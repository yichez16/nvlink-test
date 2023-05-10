/*
* Author: Yicheng Zhang
* Association: UC Riverside
* Date: May 9, 2023
*
* Description: 
* Idea of main.cu: verify coalescing behavior at nvlink level. if coalescing happens, how does coalescing happens? what is the size?
*/

// #include <vector>
// #include <cuda_profiler_api.h> // For cudaProfilerStart() and cudaProfilerStop()
// #include <cstdio>
// #include <string>
// #include <thrust/device_vector.h>
// #include <fstream>
// #include <cupti_profiler.h>
// #include <time.h>
// #include <sys/time.h>
// #include <unistd.h>
// #include <stdio.h>
// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
// #include <stdlib.h>
// #include "kernel.cu"



#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define N 1000000

__global__ void chase(int *data, int size) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < 128; k++) {
      d = data[d];
    }
  }

  data[0] = d;
}

/* simple class for a pseudo-random generator producing
   uniformely distributed integers */
class UniformIntDistribution {
public:
  UniformIntDistribution() : engine(std::random_device()()) {}
  /* return number in the range of [0..upper_limit) */
  unsigned int draw(unsigned int upper_limit) {
    return std::uniform_int_distribution<unsigned int>(0,
                                                       upper_limit - 1)(engine);
  }

private:
  std::mt19937 engine;
};

/* create a cyclic pointer chain that covers all words
   in a memory section of the given size in a randomized order */
void create_random_chain(int **indices, int len) {
  UniformIntDistribution uniform;

  // shuffle indices
  for (int i = 0; i < len; ++i) {
    (*indices)[i] = i;
  }
  for (int i = 0; i < len - 1; ++i) {
    int j = i + uniform.draw(len - i);
    if (i != j) {
      std::swap((*indices)[i], (*indices)[j]);
    }
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cout << "./main [working set size]" << std::endl;
    exit(1);
  }

  int working_set_num_size = std::stoi(argv[1]);
  int working_set_num = working_set_num_size / sizeof(int);

  int *data = new int[working_set_num];
  int **data_p = &data;

  create_random_chain(data_p, working_set_num);

  int *d_data;
  cudaMalloc(&d_data, working_set_num_size);
  cudaMemcpy(d_data, data, working_set_num_size, cudaMemcpyHostToDevice);

  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  chase<<<1, 1>>>(d_data, working_set_num);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
}