# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks

# Setup
Install [the docker](https://www.docker.com/). Here are some guides for [install docker for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and [docker for MacOS](https://docs.docker.com/desktop/install/mac-install/)

Clone this repo to your local workspace:

    $ git clone https://github.com/YUNBLAK/standalone_16bits_nn.git

This contains a Dockerfile that can be used to build our implementation. If an error about permission denied occurs, try to run docker in root sudo docker ....

    $ sudi docker build -t nn16 .

It may takes a few minutes for installing necessary packages and libraries including TensorFlow.

# Usage

## Testing
Please replace run commands with below information.

### Precision Type for Code Testing [Precision Type]:
    float16 -> Floating Point 16-bits
    float32 -> Floating Point 32-bits
    mixed_float16 -> Mixed Precision
    bfloat16 -> Bfloat16

### CNN Architecture [Model Name]:
    alex -> AlexNet
    vgg -> VGG16
    res -> ResNet
    mob -> MobileNetV2
    dense -> DenseNet
    vit -> Vision Transformer

### Layer Numbers for CNN Architectures [Number of Layers]:
    32, 56, 110, 156 for ResNet
    121, 169 for DenseNet
    8, 12, 16 for Vision Transformer (VIT)
    
## Docker

### Docker Commands:   

GPU Setting

    # If you did not install nvidia driver
    sudo apt update
    sudo apt install nvidia-driver-xxx

    # Add GPG key and repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    
    # Update package list and install nvidia-docker2
    sudo apt-get update
    sudo apt-get install -y nvidia-docker2
    
    # restart docker
    sudo systemctl restart docker

Build

    docker build -t nn16 .

Run

    docker run --gpus all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

Ex.

    docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
