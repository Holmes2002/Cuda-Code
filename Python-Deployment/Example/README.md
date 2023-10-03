#### Hi, This is example for using cuda code for python

#### Fistly, create file .so
```
nvcc -shared -o add_arrays.so add_arrays.cu --compiler-options '-fPIC' -lcudart -lstdc++
```

#### Python code using ctypes: Write Python code that loads and uses the compiled CUDA code using file .so:
```
python add_arrays.py
```
