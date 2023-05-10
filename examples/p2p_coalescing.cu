/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: May 1, 2023
*
* Description: 
* Idea of p2p.cu: copy mem from remote gpu to local gpu
// Two cases:
// Case 1: using cudaMemcpyPeer() to copy d_B to local GPU. d_A on local GPU, d_B on remote GPU 
// Case 2: using vecAdd kernel to copy d_B to local GPU. d_A and d_C on local GPU, d_B on remote GPU
* The value of d_A, d_B, d_C vector are set to be 1, 2 and 100.
*/

#include <vector>
#include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()
#include <cstdio>
#include <string>
#include <thrust/device_vector.h>
#include <fstream>
#include <cupti_profiler.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "kernel.cu"




 
int main(int argc, char **argv) {

    using namespace std;
    // int numGPUs;

    int src=0; 
    int det=1;
    int sizeElement;
    int *h_A, *h_B, *h_C;
    int *d_A, *d_B, *d_C;

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    struct timeval t1, t2;


    sizeElement = atoi(argv[1]);
    printf("%d\n", sizeElement);

    size_t size = sizeElement * sizeof(int);


    // Allocate input vectors h_A, h_B and h_C in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize input vectors
    initVec(h_A, sizeElement, 1);
    initVec(h_B, sizeElement, 2);
    initVec(h_C, sizeElement, 100);

    // local GPU contains vec_A and vec_C
    cudaSetDevice(src);
    cudaMalloc((void**)&d_A, size);  
    cudaMalloc((void**)&d_C, size);

    // Copy vector A, C from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);


    // remote GPU contains vec_B 
    cudaSetDevice(det);
    cudaMalloc((void**)&d_B, size);



    // Copy vector B from host memory to device memory
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Make sure local GPU have access to remote GPU
    cudaSetDevice(src);
    cudaDeviceEnablePeerAccess(det, 0);  

    // int blockSize = 128;
    // int gridSize = (sizeElement + blockSize - 1) / blockSize;

    
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (sizeElement + threadsPerBlock - 1) / threadsPerBlock;
    
    // Start record time
    // cudaEventRecord(start);
    gettimeofday(&t1, 0);    
    

    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 
    

    


    // Launch vectorAdd_coalescing kernel 

    // clock_t start, middle, end;
    // double dramTime, totalTime, nvlinkTime;

    // // Launch vectorAdd_coalescing kernel 
    // start = clock(); // Start the clock
    vecAdd_coalescing_nvlink <<<8, 128>>>(d_A, d_B, d_C, sizeElement); // 56 SMs, 4*32 =  128 threads
    // middle = clock(); // middle clock
    // vecAdd_coalescing_nvlink <<<gridSize, blockSize>>>(d_A, d_B, d_C, sizeElement); // 56 SMs, 4*32 =  128 threads
    // end = clock(); // end clock


    // dramTime = (double) (middle - start);
    // totalTime = (double) (end - middle);
    // nvlinkTime = totalTime - dramTime;

    
    // Stop profiler
    cudaProfilerStop(); 

    // Stop time record
    // cudaEventRecord(stop);
    gettimeofday(&t2, 0);

    // cudaEventSynchronize(stop);
    double milliseconds = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000.0;


    // Copy back to host memory 

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // needed for kernel 
    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost); // test peer2peer memcpy
    cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost); // 


    double mb = sizeElement * sizeof(int) / (double)1e6;
    printf("Size of data transfer (MB): %f\n", mb);
    printf("Vector V_A (original value = 1): %d\n",h_A[sizeElement-1]);
    printf("Vector V_B (original value = 2): %d\n",h_B[sizeElement-1]);
    printf("Vector V_C (original value = 100): %d\n", h_C[sizeElement-1]);
    // printf("Dram acceess,%f\n", dramTime);
    // printf("Nvlink acceess,%f\n", nvlinkTime);
    // printf("Time (ms): %f\n", milliseconds);
    // printf("Bandwith (MB/s): %f\n",mb*1e3/milliseconds);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    exit(EXIT_SUCCESS);
 }
 