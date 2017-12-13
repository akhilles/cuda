#include <iostream>
#include <math.h>
#include <cstdint>

#define THREADS_PER_BLOCK 256

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

    do {
        numThreads /= 10;
        if (numThreads == 0) numThreads = 1;
        std::cout << "Launching " << numThreads << " threads" << std::endl;
        gpuProcess<<<(numThreads + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(N, arr);
        cudaDeviceSynchronize();
        N = numThreads;
    } while(numThreads > 1);
    

    std::cout << "MAX: " << arr[0] << std::endl;

    // Free memory
    cudaFree(arr);
    return 0;
}