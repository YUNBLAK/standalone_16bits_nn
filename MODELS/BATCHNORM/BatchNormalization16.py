
# Original BatchNormalization is from TensorFlow Keras, but we modified some part.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The V2 implementation of Normalization layers."""

import tensorflow.compat.v2 as tf

from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.dtensor import utils
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import control_flow_util
from keras.utils import tf_utils

# isort: off
from tensorflow.python.ops.control_flow_ops import (
    get_enclosing_xla_context,
)
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export


class BatchNormalizationBase(Layer):

    # By default, the base class uses V2 behavior. The BatchNormalization V1
    # subclass sets this to False to use the V1 behavior.
    _USE_V2_BEHAVIOR = True

    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        renorm=False,
        renorm_clipping=None,
        renorm_momentum=0.99,
        fused=None,
        trainable=True,
        virtual_batch_size=None,
        adjustment=None,
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        if isinstance(axis, (list, tuple)):
            self.axis = axis[:]
        elif isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                "Expected an int or a list/tuple of ints for the "
                "argument 'axis', but received: %r" % axis
            )
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(
            moving_variance_initializer
        )
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.renorm = renorm
        self.virtual_batch_size = virtual_batch_size
        self.adjustment = adjustment
        if self._USE_V2_BEHAVIOR:
            if fused:
                self._raise_if_fused_cannot_be_used()
            # We leave fused as None if self._fused_can_be_used()==True, since
            # we still may set it to False in self.build() if the input rank is
            # not 4.
            elif fused is None and not self._fused_can_be_used():
                fused = False
        elif fused is None:
            fused = True
        self.supports_masking = True

        self.fused = fused
        self._bessels_correction_test_only = True
        self.trainable = trainable

        if renorm:
            renorm_clipping = renorm_clipping or {}
            keys = ["rmax", "rmin", "dmax"]
            if set(renorm_clipping) - set(keys):
                raise ValueError(
                    f"Received invalid keys for `renorm_clipping` argument: "
                    f"{renorm_clipping}. Supported values: {keys}."
                )
            self.renorm_clipping = renorm_clipping
            self.renorm_momentum = renorm_momentum

    def _raise_if_fused_cannot_be_used(self):
        """Raises a ValueError if fused implementation cannot be used.
        In addition to the checks done in this function, the input tensors rank
        must be 4 or 5. The input rank check can only be done once the input
        shape is known.
        """
        # Note the ValueErrors in this function are caught and not reraised in
        # _fused_can_be_used(). No other exception besides ValueError should be
        # raised here.

        # Currently fused batch norm doesn't support renorm. It also only
        # supports a channel dimension on axis 1 or 3 (rank=4) / 1 or 4 (rank5),
        # when no virtual batch size or adjustment is used.
        if self.renorm:
            raise ValueError(
                "Passing both `fused=True` and `renorm=True` is "
                "not supported"
            )
        axis = [self.axis] if isinstance(self.axis, int) else self.axis
        # Axis -3 is equivalent to 1, and axis -1 is equivalent to 3, when the
        # input rank is 4. Similarly, the valid axis is -4, -1, 1, 4 when the
        # rank is 5. The combination of ranks and axes will be checked later.
        if len(axis) > 1 or axis[0] not in (-4, -3, -1, 1, 3, 4):
            raise ValueError(
                "Passing `fused=True` is only supported when axis is 1 "
                "or 3 for input rank = 4 or 1 or 4 for input rank = 5. "
                "Got axis %s" % (axis,)
            )
        if self.virtual_batch_size is not None:
            raise ValueError(
                "Passing `fused=True` is not supported when "
                "`virtual_batch_size` is specified."
            )
        if self.adjustment is not None:
            raise ValueError(
                "Passing `fused=True` is not supported when "
                "`adjustment` is specified."
            )
        # TODO(reedwm): Support fp64 in FusedBatchNorm then remove this check.
        if self._compute_dtype not in ("float16", "bfloat16", "float32", None):
            raise ValueError(
                "Passing `fused=True` is only supported when the compute "
                "dtype is float16, bfloat16, or float32. Got dtype: %s"
                % (self._compute_dtype,)
            )

    def _fused_can_be_used(self):
        try:
            self._raise_if_fused_cannot_be_used()
            return True
        except ValueError:
            return False

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value

    @property
    def _param_dtype(self):
        # Raise parameters of fp16 batch norm to fp32
        if self.dtype == tf.float16 or self.dtype == tf.bfloat16:
            # print("YOU ARE 16")
            return tf.float16
        else:
            # print("YOU ARE 32")
            return self.dtype or tf.float32

    def _support_zero_size_input(self):
        if not tf.distribute.has_strategy():
            return False
        strategy = tf.distribute.get_strategy()
        # TODO(b/195085185): remove experimental_enable_get_next_as_optional
        # after migrating all users.
        return getattr(
            strategy.extended,
            "enable_partial_batch_handling",
            getattr(
                strategy.extended,
                "experimental_enable_get_next_as_optional",
                False,
            ),
        )

    def build(self, input_shape):
        self.axis = tf_utils.validate_axis(self.axis, input_shape)
        input_shape = tf.TensorShape(input_shape)
        rank = input_shape.rank

        if self.virtual_batch_size is not None:
            if self.virtual_batch_size <= 0:
                raise ValueError(
                    f"`virtual_batch_size` must be a positive integer that "
                    f"divides the true batch size of the input tensor. "
                    f"Received: virtual_batch_size={self.virtual_batch_size}"
                )
            # If using virtual batches, the first dimension must be the batch
            # dimension and cannot be the batch norm axis
            if 0 in self.axis:
                raise ValueError(
                    "When using `virtual_batch_size`, the batch dimension "
                    "must be 0 and thus axis cannot include 0. "
                    f"Received axis={self.axis}"
                )
            if self.adjustment is not None:
                raise ValueError(
                    "When using `virtual_batch_size`, adjustment cannot "
                    "be specified"
                )

        if self.fused in (None, True):
            if self._USE_V2_BEHAVIOR:
                if self.fused is None:
                    self.fused = rank in (4, 5)
                elif self.fused and rank not in (4, 5):
                    raise ValueError(
                        "Batch normalization layers with `fused=True` only "
                        "support 4D or 5D input tensors. "
                        f"Received tensor with shape: {tuple(input_shape)}"
                    )
            else:
                assert self.fused is not None
                self.fused = rank in (4, 5) and self._fused_can_be_used()

        if self.fused:
            if self.axis == [1] and rank == 4:
                self._data_format = "NCHW"
            elif self.axis == [1] and rank == 5:
                self._data_format = "NCDHW"
            elif self.axis == [3] and rank == 4:
                self._data_format = "NHWC"
            elif self.axis == [4] and rank == 5:
                self._data_format = "NDHWC"
            elif rank == 5:
                # 5D tensors that can be passed in but should not use fused
                # batch norm due to unsupported axis.
                self.fused = False
            else:
                if rank == 4:
                    raise ValueError(
                        "Unsupported axis. The use of `fused=True` is only "
                        "possible with `axis=1` or `axis=3` for 4D input "
                        f"tensors. Received: axis={tuple(self.axis)}"
                    )
                else:
                    raise ValueError(
                        "Unsupported axis. The use of `fused=True` is only "
                        "possible with `axis=1` or `axis=4` for 5D input "
                        f"tensors. Received: axis={tuple(self.axis)}"
                    )

        axis_to_dim = {x: input_shape.dims[x].value for x in self.axis}
        for x in axis_to_dim:
            if axis_to_dim[x] is None:
                raise ValueError(
                    "Input has undefined `axis` dimension. Received input "
                    f"with shape {tuple(input_shape)} "
                    f"and axis={tuple(self.axis)}"
                )
        self.input_spec = InputSpec(ndim=rank, axes=axis_to_dim)

        if len(axis_to_dim) == 1 and self.virtual_batch_size is None:
            # Single axis batch norm (most common/default use-case)
            param_shape = (list(axis_to_dim.values())[0],)
        else:
            # Parameter shape is the original shape but with 1 in all non-axis
            # dims
            param_shape = [
                axis_to_dim[i] if i in axis_to_dim else 1 for i in range(rank)
            ]
            if self.virtual_batch_size is not None:
                # When using virtual batches, add an extra dim at index 1
                param_shape.insert(1, 1)
                for idx, x in enumerate(self.axis):
                    self.axis[idx] = x + 1  # Account for added dimension
        self._param_shape = param_shape
        if self.scale:
            self.gamma = self.add_weight(
                name="gamma",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(
                name="beta",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                trainable=True,
                experimental_autocast=False,
            )
        else:
            self.beta = None

        try:
            # Disable variable partitioning when creating the moving mean and
            # variance
            if hasattr(self, "_scope") and self._scope:
                partitioner = self._scope.partitioner
                self._scope.set_partitioner(None)
            else:
                partitioner = None
            self.moving_mean = self.add_weight(
                name="moving_mean",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_mean_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                dtype=self._param_dtype,
                initializer=self.moving_variance_initializer,
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
                experimental_autocast=False,
            )

            if self.renorm:
                # In batch renormalization we track the inference moving stddev
                # instead of the moving variance to more closely align with the
                # paper.
                def moving_stddev_initializer(*args, **kwargs):
                    return tf.sqrt(
                        self.moving_variance_initializer(*args, **kwargs)
                    )

                with tf.distribute.get_strategy().extended.colocate_vars_with(
                    self.moving_variance
                ):
                    self.moving_stddev = self.add_weight(
                        name="moving_stddev",
                        shape=param_shape,
                        dtype=self._param_dtype,
                        initializer=moving_stddev_initializer,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf.VariableAggregation.MEAN,
                        experimental_autocast=False,
                    )

                # Create variables to maintain the moving mean and standard
                # deviation.  These are used in training and thus are different
                # from the moving averages above. The renorm variables are
                # colocated with moving_mean and moving_stddev.
                # NOTE: below, the outer `with device` block causes the current
                # device stack to be cleared. The nested ones use a `lambda` to
                # set the desired device and ignore any devices that may be set
                # by the custom getter.
                def _renorm_variable(name, shape, initializer="zeros"):
                    """Create a renorm variable."""
                    var = self.add_weight(
                        name=name,
                        shape=shape,
                        dtype=self._param_dtype,
                        initializer=initializer,
                        synchronization=tf.VariableSynchronization.ON_READ,
                        trainable=False,
                        aggregation=tf.VariableAggregation.MEAN,
                        experimental_autocast=False,
                    )
                    return var

                with tf.distribute.get_strategy().extended.colocate_vars_with(
                    self.moving_mean
                ):
                    self.renorm_mean = _renorm_variable(
                        "renorm_mean", param_shape, self.moving_mean_initializer
                    )
                with tf.distribute.get_strategy().extended.colocate_vars_with(
                    self.moving_stddev
                ):
                    self.renorm_stddev = _renorm_variable(
                        "renorm_stddev", param_shape, moving_stddev_initializer
                    )
        finally:
            if partitioner:
                self._scope.set_partitioner(partitioner)
        self.built = True

    def _assign_moving_average(self, variable, value, momentum, inputs_size):
        def calculate_update_delta():
            decay = tf.convert_to_tensor(1.0 - momentum, name="decay")
            if decay.dtype != variable.dtype.base_dtype:
                decay = tf.cast(decay, variable.dtype.base_dtype)
            update_delta = (variable - tf.cast(value, variable.dtype)) * decay
            if inputs_size is not None:
                update_delta = tf.where(
                    inputs_size > 0,
                    update_delta,
                    backend.zeros_like(update_delta),
                )
            return update_delta

        with backend.name_scope("AssignMovingAvg") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign_sub(calculate_update_delta(), name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign_sub(
                        variable, calculate_update_delta(), name=scope
                    )

    def _assign_new_value(self, variable, value):
        with backend.name_scope("AssignNewValue") as scope:
            if tf.compat.v1.executing_eagerly_outside_functions():
                return variable.assign(value, name=scope)
            else:
                with tf.compat.v1.colocate_with(variable):
                    return tf.compat.v1.assign(variable, value, name=scope)

    def _fused_batch_norm(self, inputs, training):
        """Returns the output of fused batch norm."""
        if self.center:
            beta = self.beta
        else:
            beta = backend.constant(
                0.0, dtype=self._param_dtype, shape=self._param_shape
            )
        if self.scale:
            gamma = self.gamma
        else:
            gamma = backend.constant(
                1.0, dtype=self._param_dtype, shape=self._param_shape
            )

        if self._support_zero_size_input():
            # Keras assumes that batch dimension is the first dimension for
            # Batch Normalization.
            input_batch_size = tf.shape(inputs)[0]
        else:
            input_batch_size = None

        use_fused_avg_updates = (
            tf.compat.v1.executing_eagerly_outside_functions()
            and isinstance(self.momentum, (float, int))
            and get_enclosing_xla_context() is None
        )
        if use_fused_avg_updates:
            exponential_avg_factor = 1.0 - self.momentum
        else:
            exponential_avg_factor = None

        def _maybe_add_or_remove_bessels_correction(variance, remove=True):
            r"""Add or remove Bessel's correction."""
            # Removes Bessel's correction if remove == True, adds it otherwise.
            # This is to be consistent with non-fused batch norm. Note that the
            # variance computed by fused batch norm is with Bessel's correction.
            # This is only used in legacy V1 batch norm tests.
            if self._bessels_correction_test_only:
                return variance
            sample_size = tf.cast(
                tf.size(inputs) / tf.size(variance), variance.dtype
            )
            if remove:
                factor = (
                    sample_size - tf.cast(1.0, variance.dtype)
                ) / sample_size
            else:
                factor = sample_size / (
                    sample_size - tf.cast(1.0, variance.dtype)
                )
            return variance * factor

        def _fused_batch_norm_training():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=_maybe_add_or_remove_bessels_correction(
                    self.moving_variance, remove=False
                ),
                epsilon=self.epsilon,
                is_training=True,
                data_format=self._data_format,
                exponential_avg_factor=exponential_avg_factor,
            )

        def _fused_batch_norm_inference():
            return tf.compat.v1.nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format,
            )

        output, mean, variance = control_flow_util.smart_cond(
            training, _fused_batch_norm_training, _fused_batch_norm_inference
        )
        variance = _maybe_add_or_remove_bessels_correction(
            variance, remove=True
        )

        training_value = control_flow_util.constant_value(training)
        if training_value or training_value is None:
            if not use_fused_avg_updates:
                if training_value is None:
                    momentum = control_flow_util.smart_cond(
                        training, lambda: self.momentum, lambda: 1.0
                    )
                else:
                    momentum = tf.convert_to_tensor(self.momentum)

            def mean_update():
                """Update self.moving_mean with the most recent data point."""
                if use_fused_avg_updates:
                    if input_batch_size is not None:
                        new_mean = control_flow_util.smart_cond(
                            input_batch_size > 0,
                            lambda: mean,
                            lambda: self.moving_mean,
                        )
                    else:
                        new_mean = mean
                    return self._assign_new_value(self.moving_mean, new_mean)
                else:
                    return self._assign_moving_average(
                        self.moving_mean, mean, momentum, input_batch_size
                    )

            def variance_update():
                """Update self.moving_variance with the most recent data
                point."""
                if use_fused_avg_updates:
                    if input_batch_size is not None:
                        new_variance = control_flow_util.smart_cond(
                            input_batch_size > 0,
                            lambda: variance,
                            lambda: self.moving_variance,
                        )
                    else:
                        new_variance = variance
                    return self._assign_new_value(
                        self.moving_variance, new_variance
                    )
                else:
                    return self._assign_moving_average(
                        self.moving_variance,
                        variance,
                        momentum,
                        input_batch_size,
                    )

            self.add_update(mean_update)
            self.add_update(variance_update)

        return output

    def _renorm_correction_and_moments(
        self, mean, variance, training, inputs_size
    ):
        """Returns the correction and update values for renorm."""
        stddev = tf.sqrt(variance + self.epsilon)
        # Compute the average mean and standard deviation, as if they were
        # initialized with this batch's moments.
        renorm_mean = self.renorm_mean
        # Avoid divide by zero early on in training.
        renorm_stddev = tf.maximum(self.renorm_stddev, tf.sqrt(self.epsilon))
        # Compute the corrections for batch renorm.
        r = stddev / renorm_stddev
        d = (mean - renorm_mean) / renorm_stddev
        # Ensure the corrections use pre-update moving averages.
        with tf.control_dependencies([r, d]):
            mean = tf.identity(mean)
            stddev = tf.identity(stddev)
        rmin, rmax, dmax = [
            self.renorm_clipping.get(key) for key in ["rmin", "rmax", "dmax"]
        ]
        if rmin is not None:
            r = tf.maximum(r, rmin)
        if rmax is not None:
            r = tf.minimum(r, rmax)
        if dmax is not None:
            d = tf.maximum(d, -dmax)
            d = tf.minimum(d, dmax)
        # When not training, use r=1, d=0.
        r = control_flow_util.smart_cond(
            training, lambda: r, lambda: tf.ones_like(r)
        )
        d = control_flow_util.smart_cond(
            training, lambda: d, lambda: tf.zeros_like(d)
        )

        def _update_renorm_variable(var, value, inputs_size):
            """Updates a moving average and weight, returns the unbiased
            value."""
            value = tf.identity(value)

            def _do_update():
                """Updates the var, returns the updated value."""
                new_var = self._assign_moving_average(
                    var, value, self.renorm_momentum, inputs_size
                )
                return new_var

            def _fake_update():
                return tf.identity(var)

            return control_flow_util.smart_cond(
                training, _do_update, _fake_update
            )

        update_new_mean = _update_renorm_variable(
            self.renorm_mean, mean, inputs_size
        )
        update_new_stddev = _update_renorm_variable(
            self.renorm_stddev, stddev, inputs_size
        )

        with tf.control_dependencies([update_new_mean, update_new_stddev]):
            out_mean = tf.identity(mean)
            out_variance = tf.identity(variance)

        return (r, d, out_mean, out_variance)

    def _calculate_mean_and_var(self, inputs, reduction_axes, keep_dims):
        return tf.nn.moments(inputs, reduction_axes, keepdims=keep_dims)

    def _moments(self, inputs, reduction_axes, keep_dims):
        mean, variance = self._calculate_mean_and_var(
            inputs, reduction_axes, keep_dims
        )
        if self._support_zero_size_input():
            input_batch_size = tf.shape(inputs)[0]
            mean = tf.where(
                input_batch_size > 0, mean, backend.zeros_like(mean)
            )
            variance = tf.where(
                input_batch_size > 0, variance, backend.zeros_like(variance)
            )
        return mean, variance

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if self._USE_V2_BEHAVIOR:
            if isinstance(training, int):
                training = bool(training)
            if not self.trainable:
                # When the layer is not trainable, it overrides the value passed
                # from model.
                training = False
        return training

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, self.compute_dtype)
        training = self._get_training_value(training)

        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping
            # the Tensor and reusing the existing batch norm implementation
            original_shape = tf.shape(inputs)
            original_shape = tf.concat(
                [tf.constant([-1]), original_shape[1:]], axis=0
            )
            expanded_shape = tf.concat(
                [
                    tf.constant([self.virtual_batch_size, -1]),
                    original_shape[1:],
                ],
                axis=0,
            )

            # Will cause errors if virtual_batch_size does not divide the batch
            # size
            inputs = tf.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = tf.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not
                # support virtual batching
                outputs = undo_virtual_batching(outputs)
            return outputs

        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            # Do all math in float32 if given 16-bit inputs for numeric
            # stability.  In particular, it's very easy for variance to overflow
            # in float16 and for safety we also choose to cast bfloat16 to
            # float32.
            
            #inputs = tf.cast(inputs, tf.float32)
            inputs = inputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis
        # is not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (
                v is not None
                and len(v.shape) != ndims
                and reduction_axes != list(range(ndims - 1))
            ):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or
        # None.
        training_value = control_flow_util.constant_value(training)
        if training_value == False:  # noqa: E712
            mean, variance = self.moving_mean, self.moving_variance
        else:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(tf.shape(inputs))
                # Adjust only during training.
                adj_scale = control_flow_util.smart_cond(
                    training, lambda: adj_scale, lambda: tf.ones_like(adj_scale)
                )
                adj_bias = control_flow_util.smart_cond(
                    training, lambda: adj_bias, lambda: tf.zeros_like(adj_bias)
                )
                scale, offset = _compose_transforms(
                    adj_scale, adj_bias, scale, offset
                )

            # Some of the computations here are not necessary when
            # training==False but not a constant. However, this makes the code
            # simpler.
            keep_dims = (
                self.virtual_batch_size is not None or len(self.axis) > 1
            )
            mean, variance = self._moments(
                tf.cast(inputs, self._param_dtype),
                reduction_axes,
                keep_dims=keep_dims,
            )

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = control_flow_util.smart_cond(
                training,
                lambda: mean,
                lambda: tf.convert_to_tensor(moving_mean),
            )
            variance = control_flow_util.smart_cond(
                training,
                lambda: variance,
                lambda: tf.convert_to_tensor(moving_variance),
            )

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and
                # moving_variance with each sub-batch. However, since the moving
                # statistics are only used during evaluation, it is more
                # efficient to just update in one step and should not make a
                # significant difference in the result.
                new_mean = tf.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = tf.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self._support_zero_size_input():
                # Keras assumes that batch dimension is the first dimension for
                # Batch Normalization.
                input_batch_size = tf.shape(inputs)[0]
            else:
                input_batch_size = None

            if self.renorm:
                (
                    r,
                    d,
                    new_mean,
                    new_variance,
                ) = self._renorm_correction_and_moments(
                    new_mean, new_variance, training, input_batch_size
                )
                r = _broadcast(tf.stop_gradient(r, name="renorm_r"))
                d = _broadcast(tf.stop_gradient(d, name="renorm_d"))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                """Compute the updates for mean and variance."""
                return self._assign_moving_average(
                    var, value, self.momentum, input_batch_size
                )

            def mean_update():
                true_branch = lambda: _do_update(self.moving_mean, new_mean)
                false_branch = lambda: self.moving_mean
                return control_flow_util.smart_cond(
                    training, true_branch, false_branch
                )

            def variance_update():
                """Update the moving variance."""

                def true_branch_renorm():
                    # We apply epsilon as part of the moving_stddev to mirror
                    # the training code path.
                    moving_stddev = _do_update(
                        self.moving_stddev, tf.sqrt(new_variance + self.epsilon)
                    )
                    return self._assign_new_value(
                        self.moving_variance,
                        backend.relu(
                            moving_stddev * moving_stddev - self.epsilon
                        ),
                    )

                if self.renorm:
                    true_branch = true_branch_renorm
                else:
                    true_branch = lambda: _do_update(
                        self.moving_variance, new_variance
                    )

                false_branch = lambda: self.moving_variance
                return control_flow_util.smart_cond(
                    training, true_branch, false_branch
                )

            self.add_update(mean_update)
            self.add_update(variance_update)

        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(
            inputs,
            _broadcast(mean),
            _broadcast(variance),
            offset,
            scale,
            self.epsilon,
        )
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            "axis": self.axis,
            "momentum": self.momentum,
            "epsilon": self.epsilon,
            "center": self.center,
            "scale": self.scale,
            "beta_initializer": initializers.serialize(self.beta_initializer),
            "gamma_initializer": initializers.serialize(self.gamma_initializer),
            "moving_mean_initializer": initializers.serialize(
                self.moving_mean_initializer
            ),
            "moving_variance_initializer": initializers.serialize(
                self.moving_variance_initializer
            ),
            "beta_regularizer": regularizers.serialize(self.beta_regularizer),
            "gamma_regularizer": regularizers.serialize(self.gamma_regularizer),
            "beta_constraint": constraints.serialize(self.beta_constraint),
            "gamma_constraint": constraints.serialize(self.gamma_constraint),
        }
        if self.renorm:
            config["renorm"] = True
            config["renorm_clipping"] = self.renorm_clipping
            config["renorm_momentum"] = self.renorm_momentum
        if self.virtual_batch_size is not None:
            config["virtual_batch_size"] = self.virtual_batch_size
        if self.adjustment is not None:
            logging.warning(
                "The `adjustment` function of this `BatchNormalization` "
                "layer cannot be serialized and has been omitted from "
                "the layer config. It will not be included when "
                "re-creating the layer from the saved config."
            )
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.layers.experimental.SyncBatchNormalization", v1=[])
class SyncBatchNormalization(BatchNormalizationBase):
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        if kwargs.pop("fused", None):
            raise ValueError(
                "`fused` argument cannot be True for SyncBatchNormalization."
            )
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            fused=False,
            **kwargs,
        )

    def _calculate_mean_and_var(self, x, axes, keep_dims):
        with backend.name_scope("moments"):
            y = tf.cast(x, tf.float16) if x.dtype == tf.float16 else x
            replica_ctx = tf.distribute.get_replica_context()
            if replica_ctx:
                local_sum = tf.reduce_sum(y, axis=axes, keepdims=True)
                local_squared_sum = tf.reduce_sum(
                    tf.square(y), axis=axes, keepdims=True
                )
                batch_size = tf.shape(y)[axes[0]]
                y_sum = replica_ctx.all_reduce(
                    tf.distribute.ReduceOp.SUM, local_sum
                )
                y_squared_sum = replica_ctx.all_reduce(
                    tf.distribute.ReduceOp.SUM, local_squared_sum
                )
                global_batch_size = replica_ctx.all_reduce(
                    tf.distribute.ReduceOp.SUM, batch_size
                )

                axes_vals = [
                    (tf.shape(y))[axes[i]] for i in range(1, len(axes))
                ]
                multiplier = tf.reduce_prod(axes_vals)
                multiplier = multiplier * global_batch_size

                mean = y_sum / multiplier
                y_squared_mean = y_squared_sum / multiplier
                # var = E(x^2) - E(x)^2
                variance = y_squared_mean - tf.square(mean)
            else:
                mean = tf.reduce_mean(y, axes, keepdims=True, name="mean")
                
                variance = tf.reduce_mean(
                    tf.math.squared_difference(y, tf.stop_gradient(mean)),
                    axes,
                    keepdims=True,
                    name="variance",
                )
            if not keep_dims:
                mean = tf.squeeze(mean, axes)
                variance = tf.squeeze(variance, axes)
            if x.dtype == tf.float16:
                return (
                    tf.cast(mean, tf.float16),
                    tf.cast(variance, tf.float16),
                )
            else:
                return (mean, variance)


@keras_export("keras.layers.BatchNormalization", v1=[])
class BatchNormalization(BatchNormalizationBase):
    _USE_V2_BEHAVIOR = True

    @utils.allow_initializer_layout
    def __init__(
        self,
        axis=-1,
        momentum=0.99,
        epsilon=1e-3,
        center=True,
        scale=True,
        beta_initializer="zeros",
        gamma_initializer="ones",
        moving_mean_initializer="zeros",
        moving_variance_initializer="ones",
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
        **kwargs,
    ):
        super().__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            **kwargs,
        )
