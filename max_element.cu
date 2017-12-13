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

    double *h_arr = new double[N];
    for (int i = 0; i < N; i++) {
        double r = rand()/1000000.0;
        h_arr[i] = r;
    }

    double *d_arr;
    cudaMalloc(&d_arr, sizeof(double)*N);
    cudaMemcpy(d_arr, h_arr, N*sizeof(double), cudaMemcpyHostToDevice);
  

    int numThreads = N;
    int threadsPerBlock = 256;

    do {
        numThreads = N/16;
        if (numThreads == 0) numThreads = 1;
        if (numThreads < threadsPerBlock) threadsPerBlock = numThreads;
        int numBlocks = (numThreads + threadsPerBlock - 1)/threadsPerBlock;
        gpuProcess<<<numBlocks, threadsPerBlock>>>(N, d_arr);
        std::cout << "Launching " << numThreads << " threads: " << numBlocks << " blocks and " << threadsPerBlock << " threads/block" << std::endl;
        cudaDeviceSynchronize();

        N = numBlocks * threadsPerBlock;
    } while(numThreads > 1);
    
    cudaMemcpy(h_arr, d_arr, N*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "MAX: " << h_arr[0] << std::endl;

    // Free memory
    cudaFree(d_arr);
    return 0;
}