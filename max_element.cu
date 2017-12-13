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

    int N = 5000000;
    double *arr;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&arr, N*sizeof(double));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        double r = (rand()/(double)RAND_MAX) * 1000.0;
        arr[i] = r;
    }

    initVars<<<1, 1>>>();
    cudaDeviceSynchronize();
    // Run kernel on 1M elements on the GPU
    gpuProcess<<<1, 1>>>(N, arr);
    cudaDeviceSynchronize();

    std::cout << "MAX: " << arr[0] << std::endl;

    // Free memory
    cudaFree(arr);
    return 0;
}