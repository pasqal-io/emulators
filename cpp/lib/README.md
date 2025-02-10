To build the cuda kernels, from the root of the repository:

```
nvcc -Icpp/include -shared -Xcompiler -fPIC -Xcompiler -std=c++17 -Xcompiler -O3 -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=0  -O3 --dopt=on cpp/src/cuda_kernels.cu -o cpp/lib/libemulators_cuda_kernels.so
```
