# Standalone 16-bit: Missing Study for Hardware-Constrained Neural Networks
Standalone 16-bits Neural Networks



## Dependencies:


Used CUDA Version: 

    
## Docker

### Docker Commands:   
Build

    docker build -t nn16 .

Run

    docker run -p 3000:3000 nn16 [Model Name] [Precision Type] [Learning Rate] [Batch Size] -rs [Random Seed] -rn [Number of Layers]

Ex.

    docker run -p 3000:3000 nn16 res float16 0.01 256 -rs 777 -rn 56
