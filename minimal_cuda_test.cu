// minimal_cuda_test.cu
#include <cuda_runtime.h>
#include <iostream>

__global__ void kernel() {
    // Empty kernel
}

int main() {
    kernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    } else {
        std::cout << "CUDA code ran successfully!" << std::endl;
    }
    return 0;
}
