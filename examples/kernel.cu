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

// copy data from remote to local
__global__ void copyKernel(int* local, int* remote, int size)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < size)
    {
        local[tid] = remote[tid];
    }
}