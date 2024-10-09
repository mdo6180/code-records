#include <stdio.h>

#define N 2048 * 2048 // Number of elements in each vector


__global__ void initialize(int num, int *vector) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < N; i += stride) {
        vector[i] = num;
    }
}

/*
 * Optimize this already-accelerated codebase. Work iteratively,
 * and use nsys to support your work.
 *
 * Aim to profile `saxpy` (without modifying `N`) running under
 * 200,000 ns.
 *
 * Some bugs have been placed in this codebase for your edification.
 */

__global__ void saxpy(int * a, int * b, int * c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        c[i] = 2 * a[i] + b[i];
    }
}

int main()
{
    int deviceId;
    int numberOfSMs;
    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    int *a, *b, *c;

    int size = N * sizeof (int);        // The total number of bytes per vector

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // technique 1: use async prefetch to reduce page faults when copying memory from host to device.
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    // technique 2: set number of blocks to be a multiple of number of streaming multiprocessors (might not always speed up performance)
    int threads_per_block = 128;
    int number_of_blocks = 32 * numberOfSMs;

    // technique 3: initialize memory using the kernel
    initialize <<< number_of_blocks, threads_per_block >>> (2, a);
    initialize <<< number_of_blocks, threads_per_block >>> (1, b);
    initialize <<< number_of_blocks, threads_per_block >>> (0, c);

    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );

    cudaDeviceSynchronize();

    // technique 4: prefetch the results to reduce page faults when copying memory from device to host.
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    cudaFree( a ); cudaFree( b ); cudaFree( c );
}
