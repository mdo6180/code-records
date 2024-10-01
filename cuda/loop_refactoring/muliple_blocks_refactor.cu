#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop()
{
    int index = threadIdx.x + (blockDim.x * blockIdx.x);
    printf("This is iteration number %d\n", index);
}

int main()
{
  /*
   * try these other configs:
   */

  // loop<<<10, 1>>>();
  loop<<<5, 2>>>();
  // loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}

