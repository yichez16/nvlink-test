#include <cuda_runtime.h>
#include <iostream>
#include "kernel.cu"
#include <stdio.h>
#include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()

#define ARRAY_SIZE 1000000 // l2 cache size = 4MB with 128 Bytes cache line size



int main(int argc, char **argv)
{
    using namespace std;
    int secondThread;
    secondThread = atoi(argv[1]);

    int* devArrayLocal;
    int* devArrayRemote;
    const int numElements = ARRAY_SIZE * sizeof(int);
    int hostArrayLocal[ARRAY_SIZE];
    int hostArrayRemote[ARRAY_SIZE];

    // Initialize hostArrayLocal and hostArrayRemote with sequential values
    for (int i = 0; i < ARRAY_SIZE; i++)
    {
        hostArrayLocal[i] = i;
        hostArrayRemote[i] = i+1;
    }

    // Allocate memory for the arrays on the local GPU 0
    cudaSetDevice(0);
    cudaMalloc(&devArrayLocal, numElements);

    // Allocate memory for the array on the remote GPU 1
    cudaSetDevice(1);
    cudaMalloc(&devArrayRemote, numElements);

    // Copy hostArrayLocal to devArrayLocal on the local GPU 0
    cudaSetDevice(0);
    cudaMemcpy(devArrayLocal, hostArrayLocal, numElements, cudaMemcpyHostToDevice);

    // Copy hostArrayRemote to devArrayRemote on the remote GPU 1
    cudaSetDevice(1);
    cudaMemcpy(devArrayRemote, hostArrayRemote, numElements, cudaMemcpyHostToDevice);

    // Launch the kernel on the local GPU 0 to perform operations on the local array
    cudaSetDevice(0);
    arrayToL2Cache<<<1, 32>>>(devArrayLocal, ARRAY_SIZE);

    // Launch the kernel on the remote GPU 1 to perform operations on the remote array
    cudaSetDevice(1);
    arrayToL2Cache<<<1, 32>>>(devArrayRemote, ARRAY_SIZE);

    // Synchronize the local GPU 0 to ensure the kernel execution is completed
    cudaSetDevice(0);
    cudaDeviceSynchronize();

    // Start profiler // nvprof --profile-from-start off
    cudaProfilerStart(); 

    // Make sure local gpu 0 can acess remote gpu 1
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0); 


    copyKernel_single <<<1, 1>>>(devArrayLocal, devArrayRemote, 0);
    copyKernel_single <<<1, 1>>>(devArrayLocal, devArrayRemote, secondThread);
    // copyKernel_two <<<1, 2>>>(devArrayLocal, devArrayRemote, 0, 1);


    // Stop profiler
    cudaProfilerStop(); 



    // Copy devArrayLocal back to hostArrayLocal on the local GPU 0
    cudaMemcpy(hostArrayLocal, devArrayLocal, numElements, cudaMemcpyDeviceToHost);

    // Copy devArrayRemote back to hostArrayRemote on the remote GPU 1
    cudaSetDevice(1);
    cudaMemcpy(hostArrayRemote, devArrayRemote, numElements, cudaMemcpyDeviceToHost);




    // Print the modified values from both local and remote arrays
    // std::cout << "Local Array: ";
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << hostArrayLocal[i] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "Remote Array: ";
    // for (int i = 0; i < 10; i++)
    // {
    //     std::cout << hostArrayRemote[i] << " ";
    // }
    // std::cout << std::endl;

    // Free the allocated memory
    cudaFree(devArrayRemote);
    cudaFree(devArrayLocal);

    return 0;
}