#include <stdio.h>
#include <unistd.h>

__global__ void printNumber(int number)
{
    printf("%d\n", number);
}

int main()
{
    
    for (int i = 0; i < 5; ++i)
    {
        cudaStream_t stream;                    // CUDA streams are of type `cudaStream_t`.
        cudaStreamCreate(&stream);              // Note that a pointer must be passed to `cudaCreateStream`.

        // `stream` is passed as 4th argument in the execution configuration. 
        // 3rd argument of the execution configuration is the number of bytes in shared memory to be dynamically allocated per block 
        printNumber<<<1, 1, 0, stream>>>(i);    

        cudaStreamDestroy(stream);              // Note that a value, not a pointer, is passed to `cudaDestroyStream`.
    }

    cudaDeviceSynchronize();
}