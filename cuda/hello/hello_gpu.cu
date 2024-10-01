#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

__global__ void helloGPU()
{
  printf("Hello from the GPU.\n");
}

int main()
{

  helloCPU();

  // launch the kernel
  helloGPU<<<1,1>>>();

  /*
   * synchronize on the completion of the
   * `helloGPU` kernel completion before continuing the CPU
   * thread.
   */
  cudaDeviceSynchronize();

  printf("CPU main thread continues\n");
}

