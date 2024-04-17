# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks
## Abstract
Reducing the number of bits needed to encode the weights and activations of neural networks is highly desirable as it speeds up their training and inference time while reducing memory consumption. It is unsurprising that considerable attention has been drawn to developing neural networks that employ lower-precision computation. This includes IEEE 16-bit, Google bfloat16, 8-bit, 4-bit floating-point or fixed-point, 2-bit, and various mixed-precision algorithms. Out of these low-precision formats, IEEE 16-bit stands out due to its universal compatibility with contemporary GPUs. This accessibility contrasts with bfloat16, which needs high-end GPUs, or other non-standard fewer-bit designs, which typically require software simulation.

This study focuses on the widely accessible IEEE 16-bit format for comparative analysis. This analysis involves an in-depth theoretical investigation of the factors that lead to discrepancies between 16-bit and 32-bit models, including a formalization of the concepts of floating-point error and tolerance to understand the conditions under which a 16-bit model can approximate 32-bit results. Contrary to literature that credits the success of noise-tolerated neural networks to regularization effects, our study—supported by a series of rigorous experiments—provides a quantitative explanation of why standalone IEEE 16-bit floating-point neural networks can perform on par with  32-bit and mixed-precision networks in various image classification tasks. Because no prior research has studied  IEEE 16-bit as a standalone floating-point precision in neural networks,  we believe our findings will have significant impacts,  encouraging the adoption of standalone IEEE 16-bit networks in future neural network applications.

### People
Corresponding Author: Prof. Zhoulai Fu   
Co-Author: Juyoung Yun, Sol Choi, Francois Rameau, Byungkon Kang

### Keyword
Neural Networks, Low-precision, Half-precision

## Dependencies
Please check requirements.txt   
** Used CUDA Version: 12.2

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

    docker run --gpus=all -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

Ex.

    docker run --gpus=all -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
