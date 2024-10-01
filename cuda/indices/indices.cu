#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{
  /*
   * in this example, 256*1024 threads will get spun up, but only thread with threadID.x = 1023 in block with blockId.x = 255 will print success
   */

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  }
}

int main()
{
  size_t grid_dim = 256;	// number of blocks in the grid
  size_t block_dim = 1024;	// number of threads per block

  printSuccessForCorrectExecutionConfiguration<<<grid_dim, block_dim>>>();
  cudaDeviceSynchronize();
}
