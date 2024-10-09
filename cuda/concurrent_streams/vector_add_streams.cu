#include <stdio.h>

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
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 2<<24;
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // technique 1: use async prefetch to reduce page faults when copying memory from host to device.
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);

    size_t threadsPerBlock;
    size_t numberOfBlocks;

    // technique 2: set number of blocks to be a multiple of number of streaming multiprocessors (might not always speed up performance)
    threadsPerBlock = 256;
    numberOfBlocks = 32 * numberOfSMs;

    cudaError_t addVectorsErr;
    cudaError_t asyncErr;

    // technique 3: create non-default streams to launch multiple kernels asynchronously to initialize memory on the device more quickly
    cudaStream_t stream3, stream4, stream0;
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream0);

    // technique 4: initialize memory using the kernel
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream3>>>(3, a, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream4>>>(4, b, N);
    initWith<<<numberOfBlocks, threadsPerBlock, 0, stream0>>>(0, c, N);

    addVectorsInto<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);

    addVectorsErr = cudaGetLastError();
    if(addVectorsErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(addVectorsErr));

    asyncErr = cudaDeviceSynchronize();
    if(asyncErr != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(asyncErr));

    // technique 5: prefetch the results to reduce page faults when copying memory from device to host.
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);

    checkElementsAre(7, c, N);

    // destroy streams when no longer used
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream0);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}