#include <stdio.h>

/*
 * Host function to initialize vector elements. This function
 * simply initializes each element to equal its index in the
 * vector.
 */

__global__
void initWith(float num, float *a, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    a[i] = num;
  }
}

/*
 * Device kernel stores into `result` the sum of each
 * same-indexed value of `a` and `b`.
 */

__global__
void addVectorsInto(float *result, float *a, float *b, int N)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = index; i < N; i += stride)
  {
    result[i] = a[i] + b[i];
  }
}

/*
 * Host function to confirm values in `vector`. This function
 * assumes all values are the same `target` value.
 */

void checkElementsAre(float target, float *vector, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(vector[i] != target)
    {
      printf("FAIL: vector[%d] - %0.0f does not equal %0.0f\n", i, vector[i], target);
      exit(1);
    }
  }
  printf("Success! All values calculated correctly.\n");
}

int main()
{
    const int N = 2<<24;
    size_t size = N * sizeof(float);
    
    float *a;
    float *b;
    float *c;
    
    size_t threadsPerBlock;
    size_t numberOfBlocks;
    
    /*
    * nsys should register performance changes when execution configuration
    * is updated.
    */

    int deviceId;
    cudaGetDevice(&deviceId);                       // `deviceId` now points to the id of the currently active GPU.
    
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);      // `props` now has many useful properties about the active GPU device.

    int multiProcessorCount = props.multiProcessorCount;

    threadsPerBlock = 256;
    numberOfBlocks = multiProcessorCount * 32;
    printf("numberOfBlocks = %zu\n", numberOfBlocks);
    
    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    // Speed up initialization by launching kernels.

    initWith<<<numberOfBlocks, threadsPerBlock>>>(3, a, N);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(4, b, N);
    initWith<<<numberOfBlocks, threadsPerBlock>>>(0, c, N);
    
    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
    
    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));
    
    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));
    
    checkElementsAre(7, c, N);
    
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
