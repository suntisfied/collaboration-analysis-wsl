# minimal_cuda_test.py
import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
    try:
        x = torch.randn(1).cuda()
        print("CUDA test tensor:", x)
    except Exception as e:
        print("Error during CUDA tensor allocation:", e)
else:
    print("CUDA is not available. PyTorch cannot use the GPU.")
