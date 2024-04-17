# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks
Standalone 16-bits Neural Networks



## Dependencies:
Please check requirements.txt

Used CUDA Version: 12.2

    
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

    docker run --gpus=all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

Ex.

    docker run --gpus=all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
