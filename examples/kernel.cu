#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sys/time.h>



// warm up l2 cache 
__global__ void arrayToL2Cache(int* array, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
    {
        // Access and perform dummy operations on the array elements
        int value = array[tid];
        value *= 2;
        array[tid] = value;
    }
}

// copy data from remote to local. 
// One kernel only contains one thread.
__global__ void copyKernel_single(int* local, int* remote, int threadID)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    clock_t startClock = clock();
    if (tid == threadID)
    {
        local[tid] = remote[tid];
    }
    clock_t stopClock = clock();
    clock_t elapsedTime = stopClock - startClock;
    printf("ThreadID: %d,Elapsed Time: %llu cycles\n", threadID, elapsedTime);


}

// copy data from remote to local. 
// One kernel only contains two thread.
__global__ void copyKernel_two(int* local, int* remote, int threadID1, int threadID2)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    clock_t startClock = clock();
    if (tid == threadID1 or tid ==threadID2 )
    {
        local[tid] = remote[tid];
    }
    clock_t stopClock = clock();
    clock_t elapsedTime = stopClock - startClock;
    printf("ThreadID: %d,Elapsed Time: %llu cycles\n", threadID2, elapsedTime);

}