#include <stdio.h>
#include <cuda.h>
extern "C" {__global__ void add_arrays(float* a, float* b, float* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}
}
