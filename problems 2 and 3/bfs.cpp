// This application used an example from AMD (which shows how to set up the host application and a very a simple opencl program) as starter code.
//Copyright (c) 2010 Advanced Micro Devices, Inc. All rights reserved.
//
// A minimalist OpenCL program.
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <math.h>
#define NWITEMS 512
// A simple memset kernel
const char *source =
"__kernel void matmult( __global int *Graph,__global int *q,  __global int * prevstart, __global int * num, __global int * depth, __global int * search,  __global int * node  ) \n"
"{ \n"
" int NWITEMS = num[0]; \n"
" int cont=1;\n"
" while (node[0] < 0 && cont>0) { \n"


 " if ( q[get_global_id(0)] > -1){ \n"

" for (int i=0; i < NWITEMS; i++) { \n"
" if( Graph[get_global_id(0)*NWITEMS+i] > -1){ \n"
"  if (Graph[get_global_id(0)*NWITEMS+i] == search[0]){ \n"
"    node[0]=i; \n"
"    q[i] = get_global_id(0); \n"
" }else{ \n"
"if (q[i] == -1) { \n"
"  q[i]=get_global_id(0);\n"
"} \n"
"} \n"
" } \n"
"} \n"

"} \n"

"barrier(CLK_LOCAL_MEM_FENCE); \n"
"depth[get_group_id(0)]=depth[get_group_id(0)]+1; \n"
"cont=0;\n"
" for ( int cc=0; cc<get_num_groups(0); cc++){\n"
    " if (depth[cc] <NWITEMS){ \n"
	" cont=1; \n"
	" } \n"
	" } \n"
 "barrier(CLK_GLOBAL_MEM_FENCE); \n"

"} \n"
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
 struct node{
struct node * next;
int index;
};
struct node * top;
struct node * bot;
int rbfs(int * Graph, int search, int * q, int dim, int width, int depth){
int nwidth=0;
if(depth== dim || width == 0) {return -1;}
for (int j=0; j<width; j++){
   int v= (*top).index;
for (int i=0; i<dim; i++){
 if (Graph[v*dim+i] > -1 && q[i]<0){
  if (Graph[v*dim+i] == search){
   q[i]=v;
   return i;
  }else{
    struct node * n= (struct node *) malloc(sizeof(struct node));
 (*n).index=i;
 (*bot).next=n;
 q[i]=v;
 bot=n;
    nwidth++;
	}
 }
}
top=(*top).next;
}
//printf(" depth %i", depth);
return rbfs(Graph, search, q, dim, nwidth,depth+1);
}

int lbfs(int * Graph, int search, int * q, int dim, int width, int depth){

while(top!=NULL) {

   int v= (*top).index;
for (int i=0; i<dim; i++){
 if (Graph[v*dim+i] > -1 && q[i]<0){
  if (Graph[v*dim+i] == search){
   q[i]=v;
   return i;
  }else{
    struct node * n= (struct node *) malloc(sizeof(struct node));
 (*n).index=i;
 (*n).next=NULL;
 (*bot).next=n;
 q[i]=v;
 bot=n;
   
	}
 }
}
top=(*top).next;

depth++;
}
}
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
 cl_int * Graph = (cl_int * ) malloc(sizeof(cl_int)*NWITEMS*NWITEMS);
 cl_int * q = (cl_int * ) malloc(sizeof(cl_int)*NWITEMS);
 cl_int * prevstart = (cl_int * ) malloc(sizeof(cl_int)*NWITEMS*NWITEMS);
  srand(time(0));
 int rx[NWITEMS];
  
  for (int mx=0; mx < NWITEMS*NWITEMS; mx++){
    Graph[mx]=-1;
	}
	for (int y=0; y < NWITEMS; y++){
  rx[y]=rand()%100;
  Graph[y*NWITEMS+y]=rx[y];
  q[y]=-1;
	//prevstart[i*NWITEMS+j]=0;
  }
 for (int i=0; i<NWITEMS; i++){

    for(int j=i+1; j<NWITEMS; j++){
	
	int r2 = rand()%10;
	if (r2 > 8 ){
	Graph[i*NWITEMS+j]= rx[j];
	Graph[j*NWITEMS+i]=rx[i];
	
	}
	
	
	
	
	}
	}
	
	FILE * file=fopen( "Graph1.txt", "w");
     for (int i=0; i<NWITEMS; i++){

    for(int j=0; j<NWITEMS; j++){
	
         fprintf(file,"%i ",Graph[i*NWITEMS+j]);
	}
	fprintf(file,"\n");
	}
	fclose(file);
	/*Graph[90]=9;
	Graph[76]=9;
	Graph[90*512+17]=9;
	Graph[76*512+87]=9;
	Graph[76*512+302]=9;
	Graph[76*512+23]=63;
	Graph[17*512+57]=8;
	Graph[17*512+8]=12;
	Graph[57*512+108]=23;
	Graph[108*512+57]=23;
	Graph[87*512+99]=23;
	Graph[302*512+509]=23;
	Graph[509*512+65]=5;*/
	q[0]=0;
 int i;
 printf("graph %i %i\n", Graph[79]);
 top= (struct node *) malloc(sizeof(struct node));
 (*top).index=0;
 (*top).next=NULL;
 bot=top;
   struct timespec start, finish;
	double elapsed;
  clock_gettime(CLOCK_MONOTONIC, &start);
 int nr=rbfs(Graph, 5, q, NWITEMS, 1,0);
  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " rbfs program");
    printf("Time elapsed %f secs", elapsed );
 printf("node %i %i \n", nr, Graph[i]);
 int tx=nr;
 printf("path from node to root: ");
while (tx != 0){

 printf(" %i -> ",tx);
 tx=q[tx];

 }
 printf("0 \n");
 for(int j=0; j<NWITEMS; j++){
	
	q[j]=-1;
	
	}
	q[0]=0;
	
	q[0]=0;
 
// printf("graph %i %i\n", Graph[79]);
 top= (struct node *) malloc(sizeof(struct node));
 (*top).index=0;
 (*top).next=NULL;
 bot=top;
    
  clock_gettime(CLOCK_MONOTONIC, &start);
 nr=lbfs(Graph, 5, q, NWITEMS, 1,0);
  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " lbfs program");
    printf("Time elapsed %f secs", elapsed );
 printf("node %i %i \n", nr, Graph[i]);
  tx=nr;
 printf("path from node to root: ");
while (tx != 0){

 printf(" %i -> ",tx);
 tx=q[tx];

 }
 printf("0 \n");
 for(int j=0; j<NWITEMS; j++){
	
	q[j]=-1;
	
	}
	q[0]=0;
	
	
 cl_int * nit = (cl_int *) malloc(sizeof(cl_int));
 nit[0]=NWITEMS;
 cl_int * depth = (cl_int *) malloc(sizeof(cl_int));
 depth[0]=0;
 cl_int * search = (cl_int *) malloc(sizeof(cl_int));
 search[0]=5;
  cl_int * node = (cl_int *) malloc(sizeof(cl_int));
 node[0]=-1;
 cl_mem buffer = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, NWITEMS*NWITEMS*sizeof(cl_int), (void *) Graph, NULL);
  cl_mem buffer2 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, NWITEMS*sizeof(cl_int), (void *) q, NULL);
  cl_mem buffer3 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, NWITEMS*NWITEMS*sizeof(cl_int), (void *) prevstart, NULL);
  cl_mem buffer4 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, sizeof(cl_int), (void *) nit, NULL);
  cl_mem buffer5 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, sizeof(cl_int), (void *) depth, NULL);
  cl_mem buffer6 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, sizeof(cl_int), (void *) search, NULL);
  cl_mem buffer7 = clCreateBuffer( context,CL_MEM_USE_HOST_PTR, sizeof(cl_int), (void *) node, NULL);
 
 // 6. Launch the kernel. Let OpenCL pick the local work size.
 size_t global_work_size[1] ;
 global_work_size[0]=NWITEMS;
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

  clock_gettime(CLOCK_MONOTONIC, &start);

 err=clEnqueueNDRangeKernel( queue,kernel,1,NULL,global_work_size,NULL, 0, NULL, NULL);
 /*if ( err == CL_INVALID_KERNEL  ){
printf("invalid work group size %ui",err);
}else{
if (err == CL_SUCCESS){
printf("success");
}
}*/
 clFinish( queue );
  clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf( " opencl program");
    printf("Time elapsed %f secs", elapsed );
 // 7. Look at the results via synchronous buffer map.

 cl_int *ptr;
 ptr = (cl_int *) clEnqueueMapBuffer( queue,buffer2,CL_TRUE,CL_MAP_READ,0,NWITEMS*sizeof(cl_int),0, NULL, NULL, NULL );
cl_int *ptr2;
 ptr2 = (cl_int *) clEnqueueMapBuffer( queue,buffer5,CL_TRUE,CL_MAP_READ,0,sizeof(cl_int),0, NULL, NULL, NULL );
//printf("depth %i\n",ptr2[0]);
 cl_int *ptr3;
 ptr3 = (cl_int *) clEnqueueMapBuffer( queue,buffer7,CL_TRUE,CL_MAP_READ,0,sizeof(cl_int),0, NULL, NULL, NULL );
printf("node %i\n",ptr3[0]);
 tx=ptr3[0];
 /*for (i=0; i<NWITEMS; i++){
  if (ptr[i] > -1){
  printf("%i %i\n", i, ptr[i]);
  }
  
 }
 printf("path from node to root: ");
 */
while (tx != 0){

 printf(" %i -> ",tx);
 tx=ptr[tx];

 }
 printf("0 \n" );
 return 0;
}