import ctypes
import numpy as np

# Load the CUDA code shared library
cuda_lib = ctypes.CDLL('./add_arrays.so')

# Define the input arrays
size = 5
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
# Define the output array
result = np.zeros_like(a)

# Allocate GPU memory for input and output arrays
d_a = ctypes.c_void_p()
d_b = ctypes.c_void_p()
d_result = ctypes.c_void_p()
cuda_lib.cudaMalloc(ctypes.byref(d_a), size * ctypes.sizeof(ctypes.c_float))
cuda_lib.cudaMalloc(ctypes.byref(d_b), size * ctypes.sizeof(ctypes.c_float))
cuda_lib.cudaMalloc(ctypes.byref(d_result), size * ctypes.sizeof(ctypes.c_float))

# Copy input data to the GPU


cuda_lib.cudaMemcpy(d_a, a.ctypes.data, size * ctypes.sizeof(ctypes.c_float))

cuda_lib.cudaMemcpy(d_b, b.ctypes.data, size * ctypes.sizeof(ctypes.c_float))

# Launch the CUDA kernel
block_size = 32
grid_size = (size + block_size - 1) // block_size
print("LOAD CPT")
cuda_lib.add_arrays(d_a, d_b, d_result, size, block_size, grid_size)

# Copy the result back from the GPU
cuda_lib.cudaMemcpy(result.ctypes.data, d_result, size * ctypes.sizeof(ctypes.c_float), 2)

# Free GPU memory
cuda_lib.cudaFree(d_a)
cuda_lib.cudaFree(d_b)
cuda_lib.cudaFree(d_result)

print("Array A:", a)
print("Array B:", b)
print("Result:", result)