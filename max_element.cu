#include <iostream>
#include <math.h>
#include <cstdint>

__global__ void gpuProcess(int n, double *arr){
    double localMax = -1;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        if (arr[i] > localMax) localMax = arr[i];
    }

    arr[index] = localMax;
}

int main(void){
    int N = 50000000;
    double *arr;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&arr, N*sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        double r = rand()/1000000.0;
        arr[i] = r;
    }

    int numThreads = N;
    int threadsPerBlock = 256;

    do {
        numThreads = N/16;
        if (numThreads == 0) numThreads = 1;
        if (numThreads < threadsPerBlock) threadsPerBlock = numThreads;
        int numBlocks = (numThreads + threadsPerBlock - 1)/threadsPerBlock;
        gpuProcess<<<numBlocks, threadsPerBlock>>>(N, arr);

        std::cout << "Launching " << numThreads << " threads: " << numBlocks << " blocks and " << threadsPerBlock << " threads/block" << std::endl;

        cudaDeviceSynchronize();
        N = numBlocks * threadsPerBlock;
    } while(numThreads > 1);
    

    std::cout << "MAX: " << arr[0] << std::endl;

    // Free memory
    cudaFree(arr);
    return 0;
}