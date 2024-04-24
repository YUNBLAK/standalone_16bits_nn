# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks

# Setup
Install [the docker](https://www.docker.com/). Here are some guides for [install docker for Ubuntu](https://docs.docker.com/engine/install/ubuntu/) and [docker for MacOS](https://docs.docker.com/desktop/install/mac-install/)

Clone this repo to your local workspace:

    $ git clone https://github.com/YUNBLAK/standalone_16bits_nn.git

This contains a Dockerfile that can be used to build our implementation. If an error about permission denied occurs, try to run docker in root sudo docker ....

    $ sudi docker build -t nn16 .

It may takes a few minutes for installing necessary packages and libraries including TensorFlow.

# Usage
Run command type:

    sudo docker run --gpus all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

Examples:

    sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 32
    sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
    sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 110
    sudo docker run --gpus all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 156

    sudo docker run --gpus all -p 3000:3000 nn16 alex float16 0.01 256 -rs 777 -rn 0
    sudo docker run --gpus all -p 3000:3000 nn16 vgg float16 0.01 256 -rs 777 -rn 0

    sudo docker run --gpus all -p 3000:3000 nn16 mob float16 0.01 256 -rs 777 -rn 0

    sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 121
    sudo docker run --gpus all -p 3000:3000 nn16 dense float16 0.01 256 -rs 777 -rn 169

    sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 8
    sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 12
    sudo docker run --gpus all -p 3000:3000 nn16 vit float16 0.01 256 -rs 777 -rn 16
    
