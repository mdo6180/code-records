#include <stdio.h>

__global__ void loop()
{
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, only use 1 block of threads.
   */

  int N = 10;
  loop<<<1, N>>>();	// run the kernel with 1 block of 10 threads
  cudaDeviceSynchronize();
}

