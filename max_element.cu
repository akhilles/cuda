#include <iostream>
#include <math.h>
#include <cstdint>
#include <time.h>
#include <cstdio>

void cpuProcess(int n, double *arr){
    double localMax = -1;
    
    for (int i = 0; i < n; i ++){
        if (arr[i] > localMax) localMax = arr[i];
    }
    
    arr[0] = localMax;
}

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
    clock_t start, diff;
    int N = 200000000;

    double *h_arr = new double[N];
    for (int i = 0; i < N; i++) {
        double r = rand()/1000000.0;
        h_arr[i] = r;
    }

    start = clock();
    cpuProcess(N, h_arr);
    diff = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    std::cout << "CPU MAX: " << h_arr[0] << std::endl;
    printf("Time taken for cpu: %d milliseconds\n\n", diff);

    start = clock();
    double *d_arr;
    cudaMalloc(&d_arr, sizeof(double)*N);
    cudaMemcpy(d_arr, h_arr, N*sizeof(double), cudaMemcpyHostToDevice);
    diff = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken to copy arr to gpu: %d milliseconds\n", diff);
  
    int numThreads = N;
    int threadsPerBlock = 256;

    start = clock();
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
    
    cudaMemcpy(h_arr, d_arr, 1*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "GPU MAX: " << h_arr[0] << std::endl;
    diff = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time taken for gpu: %d milliseconds\n", diff);

    // Free memory
    cudaFree(d_arr);
    return 0;
}