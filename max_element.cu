#include <iostream>
#include <math.h>

__device__ double d_max, d_min;

__device__ void AtomicMax(double * const address, const double value){
	if (* address >= value) return
	uint64 * const address_as_i = (uint64 *)address;
    uint64 old = * address_as_i, assumed;
	do {
        assumed = old;
		if (__longlong_as_double(assumed) >= value) break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

__device__ void AtomicMin(double * const address, const double value){
	if (* address <= value) return
	uint64 * const address_as_i = (uint64 *)address;
    uint64 old = * address_as_i, assumed;
	do {
        assumed = old;
		if (__longlong_as_double(assumed) <= value) break;
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

__global__ void initVars(){
    d_max = -1;
    d_min = 1001;
}

__global__ void gpuProcess(int n, float *arr){
    double localMax = -1, localMin = 1001;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        if (arr[i] > localMax) localMax = arr[i];
        if (arr[i] < localMin) localMin = arr[i];
    }

    AtomicMax(&d_max, localMax)
    AtomicMin(&d_min, localMin)
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

    typeof(d_max) h_max;
    cudaMemcpyFromSymbol(&h_max, "d_max", sizeof(h_max), 0, cudaMemcpyDeviceToHost);
    std::cout << "MAX: " << h_max << std::endl;

    // Free memory
    cudaFree(arr);
    return 0;
}