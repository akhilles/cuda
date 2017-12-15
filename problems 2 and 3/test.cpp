// This application used an example from AMD (which shows how to set up the host application and a very a simple opencl program) as starter code.
//Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
// A minimalist OpenCL program.
#include <CL/cl.h>
#include <clBLAS.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <math.h>
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
" int h1=get_group_id(0);\n"
" int h2=get_group_id(1); \n"
" tileC[(get_local_id(0))*16 + get_local_id(1)] = 0; \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"
 "for (int hh=0; hh<NWITEMS/16; hh++){ \n"
 
 " for (int j=0; j< size1; j++) { \n"
"event_t ev[2]; \n"
" ev[0] = async_work_group_copy(&tileA[j*size2], &A[(j+h1*size1)*NWITEMS+hh*16], 16,0); \n"
" ev[1]= async_work_group_copy(&tileB[j*size2], &B[(j+hh*16)*get_global_size(1)+h2*size2], size2,0); \n"
"wait_group_events(2,ev ); \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"
"barrier(CLK_GLOBAL_MEM_FENCE); \n"
" } \n"


" float loc=0; \n"
" for (int i=0; i < 16; i++) { \n"
" loc+= tileA[get_local_id(0)*16+i]*tileB[i*size2+get_local_id(1)]; \n"
" }tileC[get_local_id(0)*size2+get_local_id(1)]+=loc; \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"
"barrier(CLK_GLOBAL_MEM_FENCE); \n"
"} \n"


" for (int j=0; j< size1; j++) { \n"
"event_t ev[1]; \n"
" ev[0] = async_work_group_copy(&C[(j+h1*size1)*get_global_size(1)+h2*size2],&tileC[(j*size2)], size2,0); \n"
"wait_group_events(1,ev ); \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"
"barrier(CLK_GLOBAL_MEM_FENCE); \n"
"} \n"


"} \n";

const char *source4 =
"__kernel void matmult( __global float *A,__global float *B,__global float *C, __global int * num) \n"
"{ \n"
 " int NWITEMS = num[0]; \n"
" __local float  tileA[64]; \n"
" __local float  tileB[64]; \n"
"__local float tileC; \n"
"tileC=0; \n"
" size_t size1=64 ;\n"
" size_t size3 = 64;  \n"
" size_t size2 =64 ; \n"
" int h1=get_global_id(0);\n"
" int h2=get_global_id(1); \n"
 "for (int hh=0; hh<NWITEMS/64; hh++){ \n"
 
"event_t ev[2]; \n"
" ev[0] = async_work_group_copy(tileA, &A[(h1)*NWITEMS+hh*64], 64,0); \n"
" ev[1]= async_work_group_strided_copy(tileB, &B[(hh)*get_global_size(1)+h2], 64,get_global_size(1),0); \n"
"wait_group_events(2,ev ); \n"

" float loc=0; \n"
" for (int i=0; i < 2; i++) { \n"
" loc+= tileA[get_global_id(2)*2+i]*tileB[get_global_id(2)*2+i]; \n"
" }tileC+=loc; \n"
"barrier(CLK_LOCAL_MEM_FENCE); \n"
"} \n"

 

" C[(h1)*get_global_size(1)+h2] =tileC; \n"



"} \n";
void matmult(cl_float * A, cl_float * B, cl_float * C, size_t n1, size_t n2, size_t n3){

 for (size_t i=0; i<n1; i++){
    for (size_t j=0; j<n3; j++){
	  for(size_t q=0; q<n2; q++){
	     C[i*n3+j]+=A[i*n2+q]*B[q*n3+j];
		 
	  }
	
	}
  }


}

void blas(cl_float * A, cl_float * B, cl_float * C, size_t n1, size_t n2, size_t n3){

cl_platform_id platform;

 clGetPlatformIDs( 1, &platform, NULL );
 // 2. Find a gpu device.
 cl_device_id device;
const char * sourc=source;
 
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
 clblasSetup( );
//printf("setup \n");
 cl_float alpha=1;
 cl_float beta=1;
 int i;
//for(i=0; i < NWITEMS-500; i++)
 //printf("%f %f\n", i, A[i]);
 cl_int * nit = (cl_int *) malloc(sizeof(cl_int));
 nit[0]=NWITEMS;
 cl_int err;
 cl_mem buffer = clCreateBuffer( context,CL_MEM_READ_ONLY, n1*n2*sizeof(cl_float), NULL, &err);
 cl_mem buffer2 = clCreateBuffer( context,CL_MEM_READ_ONLY, n2*n3*sizeof(cl_float), NULL, &err);
 cl_mem buffer3 = clCreateBuffer( context,CL_MEM_READ_WRITE, n1*n3*sizeof(cl_float), NULL, &err);

  
 // 6. Launch the kernel. Let OpenCL pick the local work size.
 err= clEnqueueWriteBuffer( queue, buffer, CL_TRUE, 0, n1*n2*sizeof(cl_float), A, 0, NULL, NULL );
 
 err= clEnqueueWriteBuffer( queue, buffer2, CL_TRUE, 0,  n2*n3*sizeof(cl_float), B, 0, NULL, NULL );
  err= clEnqueueWriteBuffer( queue, buffer3, CL_TRUE, 0, n1*n3*sizeof(cl_float), C, 0, NULL, NULL );
  if (err != CL_SUCCESS){
  printf("error");
  }
    cl_event event = NULL;
	//printf("blas \n");
	  struct timespec start, finish;
	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &start);
   err= clblasSgemm( clblasRowMajor, clblasNoTrans, clblasNoTrans,
                                n1, n3, n2,
                                1, buffer, 0, n2,
                                buffer2, 0, n3, 1,
                                buffer3, 0, n3,
                                1, &queue, 0, NULL, &event );
			clWaitForEvents( 1, &event );
			 clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
		if (err != clblasSuccess){
		printf("error bls");
		}
		 printf("clBLAS Time elapsed %f secs\n", elapsed );
/* if ( err == CL_INVALID_KERNEL  ){
printf("invalid work group size %ui",err);
}else{
if (err == CL_SUCCESS){
//printf("success");
}
}*/
cl_float *ptr = (cl_float *) malloc(sizeof(cl_float)*n1*n2);

 clEnqueueReadBuffer( queue,buffer3,CL_TRUE,0,n1 *n3 *sizeof(cl_float),C,0, NULL, NULL );




}
int main(int argc, char ** argv)
{
printf("%s", argv[1]);

// 5. Create a data buffer.
 size_t n1 = 64*4;
 size_t n2= 32*4;
 cl_float * A = (cl_float * ) malloc(sizeof(cl_float)*n1*NWITEMS);
 cl_float * B = (cl_float * ) malloc(sizeof(cl_float)*NWITEMS*n2);
 cl_float * C = (cl_float * ) malloc(sizeof(cl_float)*n1*n2);
 cl_float * res = (cl_float * ) malloc(sizeof(cl_float)*n1*n2);
 srand(time(0));
 for (int i=0; i<n1; i++){
    for(int j=0; j<NWITEMS; j++){
	//A[i*NWITEMS+j]= ((i+1)*NWITEMS)/(j+23);
	 float r1=rand();
	float r2=rand();
	A[i*NWITEMS+j]=r1/r2;
	
	}
	}
	for (int i=0; i<NWITEMS; i++){
    for(int j=0; j<n2; j++){
	float r1=rand();
	float r2=rand();
	B[i*n2+j]= r1/r2;
	
	}
	}
 int i;
 for (int mn=0; mn < n1*n2; mn++){
 C[mn]=0;
 res[mn]=0;
 }
 
  blas(A,B,res, n1,NWITEMS,n2);
  for(i=0; i < 4; i++)
 printf("%d %f\n", i, res[i]);
 // 1. Get a platform.
 cl_platform_id platform;

 clGetPlatformIDs( 1, &platform, NULL );
 // 2. Find a gpu device.
 cl_device_id device;
const char * sourc=source;
 if (*argv[1]=='1'){
   sourc=source;}
 else if(*argv[1] =='2'){
    sourc=source2;
 }else if(*argv[1] =='3'){
    sourc=source3;
 }else if(*argv[1] == '4'){
    sourc=source4;
 
 }
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
 &sourc,
 NULL, NULL );
 cl_ulong size;
clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
//printf("size %u ", size);
 cl_int err=clBuildProgram( program, 1, &device, NULL, NULL, NULL );
if (err != CL_SUCCESS){
printf("build error");
}else{
//printf("success");
}
 cl_kernel kernel = clCreateKernel( program, "matmult", NULL );
 
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
  if (*argv[1] == '3'){
   err=clEnqueueNDRangeKernel( queue,kernel,2,NULL,global_work_size,local_work_size, 0, NULL, NULL);
   }else if(*argv[1] =='4'){
   size_t g_work_size[3] ;
 g_work_size[0]=n1;
 g_work_size[1]=n2;
 g_work_size[2]=32;
  local_work_size[0]=1;
 local_work_size[1]=1;
 local_work_size[2]=32;
  err=clEnqueueNDRangeKernel( queue,kernel,3,NULL,g_work_size,local_work_size, 0, NULL, NULL);
}else
   {
 err=clEnqueueNDRangeKernel( queue,kernel,2,NULL,global_work_size,local_work_size, 0, NULL, NULL);
 }

/* if ( err == CL_INVALID_KERNEL  ){
printf("invalid work group size %ui",err);
}else{
if (err == CL_SUCCESS){
//printf("success");
}
}*/
 clFinish( queue );

  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
     if (*argv[1]=='1'){
   printf("global memory opencl ");}
 else if(*argv[1] =='2'){
    printf("global memory opencl loop unrolling ");
 }else if(*argv[1] =='3'){
     printf("matrix tiling local memory opencl ");
 }else if(*argv[1] == '4'){
     printf("local memory opencl\n ");
 
 }
    printf("Time elapsed %f secs\n", elapsed );
	
	/* clock_gettime(CLOCK_MONOTONIC, &start);
	 cl_float X[6];
	 cl_float Y[6];
	 cl_float Z[4];
	 X[0]=1;
	 X[1]=2;
	 X[2]=3;
	 X[3]=4;
	  X[4]=5;
	 X[5]=6;
	 
	 Y[0]=1;
	 Y[1]=2;
	 Y[2]=3;
	 Y[3]=4;
	 Y[4]=5;
	 Y[5]=6;
	 Z[0]=0;
	 Z[1]=0;
	 Z[2]=0;
	 Z[3]=0;
	 blas(A,B,C, n1,NWITEMS,n2);
	matmult(A,B,C, n1,NWITEMS,n2);
	printf("----------\n");
	for(i=0; i < 4; i++)
printf("%d %f\n", i, Z[i]);
printf("----------\n");
	  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " serial program\n");
    printf("Time elapsed %f secs\n", elapsed );
	*/
	//for(i=0; i < 30; i++)
   //printf("%f %f\n", i, C[i]);
	//printf("--------------------");
 // 7. Look at the results via synchronous buffer map.
 cl_float *ptr;
 ptr = (cl_float *) clEnqueueMapBuffer( queue,buffer3,CL_TRUE,CL_MAP_READ,0,n1 *n2 *sizeof(cl_float),0, NULL, NULL, NULL );

 for(i=0; i < 4; i++)
  printf("%d %f\n", i, ptr[i]);
 float resx=0;
for(i=0; i < n1*n2; i++)
{
resx=resx+(ptr[i]-res[i])*(ptr[i]-res[i]);
if ((ptr[i]-res[i])*(ptr[i]-res[i]) >10){
//printf("err %i %f \n ", i, res[i]);
}
}
printf("residual %f \n", resx);


	 cl_float X[6];
	 cl_float Y[6];
	 cl_float Z[4];
	 X[0]=1;
	 X[1]=2;
	 X[2]=3;
	 X[3]=4;
	  X[4]=5;
	 X[5]=6;
	 
	 Y[0]=1;
	 Y[1]=2;
	 Y[2]=3;
	 Y[3]=4;
	 Y[4]=5;
	 Y[5]=6;
	 Z[0]=0;
	 Z[1]=0;
	 Z[2]=0;
	 Z[3]=0;
	  for (int mn=0; mn < n1*n2; mn++){
 C[mn]=0;
 res[mn]=0;
 }
	 clock_gettime(CLOCK_MONOTONIC, &start);
	matmult(A,B,C, n1,NWITEMS,n2);
	  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " serial program ");
    printf("Time elapsed %f secs\n", elapsed );
	
	 for(i=0; i < 4; i++)
 printf("%d %f\n", i, C[i]);
 return 0;
}