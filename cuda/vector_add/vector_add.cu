#include <stdio.h>

void initWith(float num, float *a, int N) {
  for(int i = 0; i < N; ++i) {
    a[i] = num;
  }
}

__global__ void addVectorsInto(float *result, float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for(int i = idx; i < N; i += grid_stride) {
        result[i] = a[i] + b[i];
    }
}

void checkElementsAre(float target, float *array, int N) {
    for(int i = 0; i < N; i++) {
        if(array[i] != target) {
            printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
            exit(1);
        }
    }
    printf("SUCCESS! All values added correctly.\n");
}

int main() {
    const int N = 2<<20;
    size_t size = N * sizeof(float);

    float *a;
    float *b;
    float *c;

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    initWith(3, a, N);
    initWith(4, b, N);
    initWith(0, c, N);

    size_t threads_per_block = 256;
    size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;
    printf("threads per block = %d\n", threads_per_block);
    printf("number of blocks = %d\n", number_of_blocks);

    addVectorsInto<<<number_of_blocks, threads_per_block>>>(c, a, b, N);
    cudaDeviceSynchronize();

    checkElementsAre(7, c, N);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}