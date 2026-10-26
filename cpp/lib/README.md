To build the cuda kernels, from the root of the repository:

```
nvcc -Icpp/include -shared -Xcompiler -fPIC -std=c++20 -Xcompiler -O3 -Xcompiler -D_GLIBCXX_USE_CXX11_ABI=1  -O3 --dopt=on -arch=native cpp/src/cuda_kernels.cu -o cpp/lib/libemulators_cuda_kernels.so
```
