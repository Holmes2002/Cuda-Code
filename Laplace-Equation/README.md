Hi, this is code for solve Laplace Equation by Gauss-Seidel Iterative

### Running

```
%%bash
nvcc main.cu
./a.out 0.0001 "result.txt"
```

### Visualize 
```
python draw_heat_map.py
````

### Experiments
#### Results on CPU and GPU 
<p align="center">
<img src="../fig/Fig_1.png" width="300" title="hover text">
</p>

#### Heat map 
<p align="center">
<img src="../fig/FIg_2.png" width="400" title="hover text">
</p>
