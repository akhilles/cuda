// This application used an example from AMD (which shows how to set up the host application and a very a simple opencl program) as starter code.
//Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
// A minimalist OpenCL program.
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#define NWITEMS 512*2
// A simple memset kernel
const char *source =
"__kernel void matmult( __global float *A,__global float *B,__global float *C, __global int * num  ) \n"
"{ \n"
" float loc=0; \n"
" int NWITEMS = num[0]; \n"
" for (int i=0; i < NWITEMS; i++) { \n"
" loc += A[get_global_id(0)*NWITEMS+i]*B[i*get_global_size(1)+get_global_id(1)]; \n"
" } \n"
" C[get_global_id(0)*get_global_size(1)+get_global_id(1)]=loc; \n"
"} \n";

const char *source2 =
"__kernel void matmult( __global float *A,__global float *B,__global float *C, __global int * num  ) \n"
"{ \n"
" float loc=0; \n"
" int NWITEMS = num[0]; \n"
" for (int i=0; i+3 < NWITEMS; i+=4) { \n"
" loc += A[get_global_id(0)*NWITEMS+i]*B[i*get_global_size(1)+get_global_id(1)]; \n"
" loc += A[get_global_id(0)*NWITEMS+(i+1)]*B[(i+1)*get_global_size(1)+get_global_id(1)]; \n"
" loc += A[get_global_id(0)*NWITEMS+(i+2)]*B[(i+2)*get_global_size(1)+get_global_id(1)]; \n"
" loc += A[get_global_id(0)*NWITEMS+(i+3)]*B[(i+3)*get_global_size(1)+get_global_id(1)]; \n"
" } \n"
" C[get_global_id(0)*get_global_size(1)+get_global_id(1)]=loc; \n"
"} \n";

const char *source3 =
"__kernel void matmult( __global float *A,__global float *B,__global float *C, __global int * num) \n"
"{ \n"
 " int NWITEMS = num[0]; \n"
" __local float  tileA[16*16]; \n"
" __local float  tileB[16*16]; \n"
"__local float tileC[16*16]; \n"
" size_t size1=16 ;\n"
" size_t size3 = 16;  \n"
" size_t size2 =16 ; \n"
" int h1=get_global_id(0)/16;\n"
" int h2=get_global_id(1)/16; \n"
 "for (int hh=0; hh<NWITEMS/16; hh++){ \n"
 " for (int j=0; j< size1; j++) { \n"
"event_t ev[2]; \n"
" ev[0] = async_work_group_copy(&tileA[j*size2], &A[(j+h1*size1)*NWITEMS+hh*16], 16,0); \n"
" ev[1]= async_work_group_copy(&tileB[j*size2], &B[(j+hh*16)*get_global_size(1)+h2*size2], size2,0); \n"
"wait_group_events(2,ev ); \n"

" } \n"
" float loc=0; \n"
" for (int i=0; i < 16; i++) { \n"
" loc+= tileA[get_global_id(0)%(16)*16+i]*tileB[i*size2+get_global_id(1)%(size2)]; \n"
" }tileC[get_global_id(0)%(size1)*size2+get_global_id(1)%(size2)]+=loc; \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"

"} \n"

 " for (int j=0; j< size1; j++) { \n"
"event_t ev[1]; \n"
" ev[0] = async_work_group_copy(&C[(j+h1*size1)*get_global_size(1)+h2*size2],&tileC[(j*size2)], size2,0); \n"
"wait_group_events(1,ev ); \n"

" } \n"

"} \n";
int main(int argc, char ** argv)
{
 // 1. Get a platform.
 cl_platform_id platform;

 clGetPlatformIDs( 1, &platform, NULL );
 // 2. Find a gpu device.
 cl_device_id device;

 clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU,
 1,
 &device,
 NULL);
 // 3. Create a context and command queue on that device.
 cl_context context = clCreateContext( NULL,
 1,
 &device,
 NULL, NULL, NULL);
 cl_command_queue queue = clCreateCommandQueue( context,
 device,
 0, NULL );
 // 4. Perform runtime source compilation, and obtain kernel entry point.
 cl_program program = clCreateProgramWithSource( context,
 1,
 &source,
 NULL, NULL );
 cl_ulong size;
clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
printf("size %u ", size);
 cl_int err=clBuildProgram( program, 1, &device, NULL, NULL, NULL );
if (err != CL_SUCCESS){
printf("build error");
}else{
printf("success");
}
 cl_kernel kernel = clCreateKernel( program, "matmult", NULL );
 // 5. Create a data buffer.
 size_t n1 = 64*2;
 size_t n2= 32*2;
 cl_float * A = (cl_float * ) malloc(sizeof(cl_float)*n1*NWITEMS);
 cl_float * B = (cl_float * ) malloc(sizeof(cl_float)*NWITEMS*n2);
 cl_float * C = (cl_float * ) malloc(sizeof(cl_float)*n1*n2);
 for (int i=0; i<n1; i++){
    for(int j=0; j<NWITEMS; j++){
	A[i*NWITEMS+j]= ((i+1)*NWITEMS)/(j+23);
	
	}
	}
	for (int i=0; i<NWITEMS; i++){
    for(int j=0; j<n2; j++){
	B[i*n2+j]= ((i+1)*n2)/(j+23);
	
	}
	}
 int i;
//for(i=0; i < NWITEMS-500; i++)
 //printf("%f %f\n", i, A[i]);
 cl_int * nit = (cl_int *) malloc(sizeof(cl_int));
 nit[0]=NWITEMS;
 cl_mem buffer = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, n1*NWITEMS*sizeof(cl_float), (void *) A, NULL);
 cl_mem buffer2 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, NWITEMS*n2*sizeof(cl_float), (void *) B, NULL);
 cl_mem buffer3 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, n1*n2*sizeof(cl_float), (void *) C, NULL);
  cl_mem buffer4 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, sizeof(cl_int), (void *) nit, NULL);
  cl_mem buffer5 = clCreateBuffer( context,CL_MEM_READ_WRITE, n1*.5*sizeof(cl_float), NULL, NULL);
 cl_mem buffer6 = clCreateBuffer( context,CL_MEM_READ_WRITE, n2*.5*sizeof(cl_float), NULL, NULL);
  cl_mem buffer7 = clCreateBuffer( context,CL_MEM_READ_WRITE, n1*n2*.25*sizeof(cl_float), NULL, NULL);
 // 6. Launch the kernel. Let OpenCL pick the local work size.
 size_t global_work_size[2] ;
 global_work_size[0]=n1;
 global_work_size[1]=n2;
 size_t local_work_size[2] ;
 local_work_size[0]=16;
 local_work_size[1]=16;
 clSetKernelArg(kernel, 0, sizeof(buffer), (void*) &buffer);
 clSetKernelArg(kernel, 1, sizeof(buffer2), (void*) &buffer2);
 clSetKernelArg(kernel, 2, sizeof(buffer3), (void*) &buffer3);
  clSetKernelArg(kernel, 3, sizeof(buffer4), (void*) &buffer4);
  clSetKernelArg(kernel, 4, sizeof(buffer5), (void*) &buffer5);
 clSetKernelArg(kernel, 5, sizeof(buffer6), (void*) &buffer6);
  clSetKernelArg(kernel, 6, sizeof(buffer7), (void*) &buffer7);
   struct timespec start, finish;
	double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
 err=clEnqueueNDRangeKernel( queue,kernel,2,NULL,global_work_size,local_work_size, 0, NULL, NULL);
 if ( err == CL_INVALID_KERNEL  ){
printf("invalid work group size %ui",err);
}else{
if (err == CL_SUCCESS){
printf("success");
}
}
 clFinish( queue );

  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " basic opencl program");
    printf("Time elapsed %f secs", elapsed );
 // 7. Look at the results via synchronous buffer map.
 cl_float *ptr;
 ptr = (cl_float *) clEnqueueMapBuffer( queue,buffer3,CL_TRUE,CL_MAP_READ,0,n1 *n2 *sizeof(cl_float),0, NULL, NULL, NULL );

//for(i=0; i < NWITEMS-500; i++)
 //printf("%d %f\n", i, ptr[i]);
 return 0;
}