# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks

The provided repository contains the necessary instructions and information to reproduce the findings of our study, "Standalone 16-bit Training: Missing Study for Hardware-Constrained Neural Networks".

# Setup
Please install [the docker](https://www.docker.com/). Here are some guidelines for [install docker for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and [docker for MacOS](https://docs.docker.com/desktop/install/mac-install/)

Clone this repository to your local workspace:

    $ git clone https://github.com/YUNBLAK/standalone_16bits_nn.git

This contains a Dockerfile that can be used to build and test our implementation. If permission denied occurs, run docker in root sudo:

    $ sudo docker build -t nn16 .

It may takes a few minutes for installing necessary packages and libraries including TensorFlow.

# Usage
## Run command

    $ sudo docker run --gpus all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

## Run Examples
Some example convolutional neural networks in CNN_MAIN.py train CIFAR-10 and show results in our paper: Float16 is faster than both Mixed Precision and Float32.

Run on example 16-bit ResNet-32:

    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 32
    # CNN model: ResNet 32
    # Learning Rate: 0.01
    # Batchsize: 256
    # Random Seed: 777
    # Precision Type: float16



Below are some examples for commands:

ResNet 32, 56, 110, 156

    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 32
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 110
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 156

AlexNet

    $ sudo docker run --gpus all -p 3000:3000 nn16 alex float16 0.01 256 -rs 777 -rn 0

VGG16, 19

    $ sudo docker run --gpus all -p 3000:3000 nn16 vgg float16 0.01 256 -rs 777 -rn 16
    $ sudo docker run --gpus all -p 3000:3000 nn16 vgg float16 0.01 256 -rs 777 -rn 19

MobileNetV2

    $ sudo docker run --gpus all -p 3000:3000 nn16 mob float16 0.01 256 -rs 777 -rn 0

DenseNet 121, 169

    $ sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 121
    $ sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 169

Vision Transformer 8, 12, 16

    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 8
    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 12
    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 16
    

## Reproduce the Results in Paper

    $ python reproduce.py


Getting the accuracies and training time (Table 5)

| Architectures (# par. million) | FP16 (Accuracy %) | FP32 (Accuracy %) | MP (Accuracy %) | FP16 (Time s) | FP32 (Time s) | MP (Time s) | Acc. Diff. FP16-FP32 | Acc. Diff. FP16-MP | Time Speedup FP32/FP16 | Time Speedup MP/FP16 |
|--------------------------------|-------------------|-------------------|-----------------|---------------|---------------|-------------|----------------------|--------------------|------------------------|-----------------------|
| AlexNet (2.09)                 | 76.0±0.3          | 75.8±0.3          | 75.9±0.3        | 96            | 174           | 150         | 0.2                  | 0.1                | 1.8x                   | 1.5x                  |
| VGG-16 (33.76)                 | 83.7±0.3          | 83.9±0.2          | 83.8±0.2        | 377           | 857           | 455         | -0.2                 | -0.1               | 2.2x                   | 1.2x                  |
| VGG-19 (38.36)                 | 83.8±0.2          | 83.9±0.3          | 83.9±0.2        | 416           | 937           | 492         | -0.1                 | -0.1               | 2.2x                   | 1.2x                  |
| ResNet-32 (0.47)               | 80.9±0.3          | 80.9±0.4          | 80.9±0.3        | 413           | 551           | 483         | 0.0                  | 0.0                | 1.3x                   | 1.2x                  |
| ResNet-56 (0.86)               | 81.6±0.4          | 81.4±0.6          | 81.5±0.6        | 677           | 905           | 795         | 0.2                  | 0.1                | 1.3x                   | 1.2x                  |
| ResNet-110 (1.78)              | 81.8±0.4          | 81.4±0.5          | 81.8±0.5        | 1256          | 1712          | 1486        | 0.4                  | 0.0                | 1.3x                   | 1.2x                  |
| DenseNet121 (7.04)             | 72.1±0.3          | 72.6±0.4          | 73.0±0.3        | 539           | 720           | 641         | -0.5                 | -0.9               | 1.3x                   | 1.2x                  |
| DenseNet169 (12.65)            | 71.7±0.5          | 72.3±0.3          | 72.1±0.3        | 724           | 966           | 812         | -0.6                 | -0.4               | 1.3x                   | 1.1x                  |
| Xception (22.96)               | 75.9±0.4          | 76.3±0.3          | 76.3±0.4        | 324           | 611           | 412         | -0.4                 | -0.4               | 1.9x                   | 1.3x                  |
| MobileNetV2 (2.27)             | 69.4±1.1          | 70.0±1.2          | 70.0±1.1        | 353           | 588           | 411         | -0.6                 | -0.6               | 1.7x                   | 1.2x                  |
| VIT-8 (2.02)                   | 71.0±0.3          | 71.3±0.3          | 71.2±0.3        | 316           | 423           | 410         | -0.3                 | -0.2               | 1.3x                   | 1.2x                  |
| VIT-12 (2.55)                  | 71.1±0.3          | 71.4±0.3          | 71.5±0.3        | 425           | 663           | 629         | -0.3                 | -0.004             | 1.6x                   | 1.5x                  |
| **Mean**                       |                   |                   |                 |               |               |             | -0.2                 | -0.2               | 1.6x                   | 1.3x                  |
