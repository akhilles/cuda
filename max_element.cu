#include <iostream>
#include <math.h>

__device__ double gMax, gMin;

__device__ void AtomicMax(double * const address, const double value)
{
	if (* address >= value)
	{
		return;
	}

	uint64 * const address_as_i = (uint64 *)address;
    uint64 old = * address_as_i, assumed;

	do 
	{
        assumed = old;
		if (__longlong_as_double(assumed) >= value)
		{
			break;
		}
		
        old = atomicCAS(address_as_i, assumed, __double_as_longlong(value));
    } while (assumed != old);
}

__global__ void initVars(){
    gMax = -1;
    gMin = 1001;
}

__global__ void gpuMax(int n, float *arr){
    double localMax = -1, localMin = 1001;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride){
        if (arr[i] > localMax) localMax = arr[i];
        if (arr[i] < localMin) localMin = arr[i];
    }
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

    // Run kernel on 1M elements on the GPU
    gpuMax<<<1, 1>>>(N, arr);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);
  
    return 0;
}