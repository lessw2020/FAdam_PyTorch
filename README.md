# FAdam_PyTorch
an implentation of FAdam (Fisher Adam) in PyTorch

Please see the official Arxiv paper:   
[FAdam: Adam is a natural gradient optimizer using
diagonal empirical Fisher information](https://arxiv.org/abs/2405.12807)

Schedule:  
1 - impl in eager PyTorch  
2 - (if torch.compile not performant) - update to fused Cuda kernel (cpp extension)  

