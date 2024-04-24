# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks

# Setup
Please install [the docker](https://www.docker.com/). Here are some guidelines for [install docker for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and [docker for MacOS](https://docs.docker.com/desktop/install/mac-install/)

Clone this repository to your local workspace:

    $ git clone https://github.com/YUNBLAK/standalone_16bits_nn.git

This contains a Dockerfile that can be used to build and test our implementation. If permission denied occurs, run docker in root sudo:

    $ sudo docker build -t nn16 .

It may takes a few minutes for installing necessary packages and libraries including TensorFlow.

# Usage
## Run command type

    $ sudo docker run --gpus all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

## Run Examples
Some example convolutional neural networks in CNN_MAIN.py train CIFAR-10 and show results in our paper: Float16 is faster than both Mixed Precision and Float32.

Run on examples CNN for IEEE Floating Point 16-bit:

ResNet

    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 32
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 110
    $ sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 156

AlexNet

    $ sudo docker run --gpus all -p 3000:3000 nn16 alex float16 0.01 256 -rs 777 -rn 0

VGG16

    $ sudo docker run --gpus all -p 3000:3000 nn16 vgg float16 0.01 256 -rs 777 -rn 0

MobileNetV2

    $ sudo docker run --gpus all -p 3000:3000 nn16 mob float16 0.01 256 -rs 777 -rn 0


    $ sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 121
    $ sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 169

    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 8
    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 12
    $ sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 16
    
