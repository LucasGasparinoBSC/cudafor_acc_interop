# ACC data on CUDA kernels

In this example, OpenACC data directives are used to create and manage data on the GPU, which is later accessed by CUDA kernels and cuBLAS routines.

## Details

This is a fairly simple example: the driver creates some array data on both host and device using OpenACC data directives, hen launches a series of kernels and a cuBLAS routine to operate on that data. The key point is the `acc host_data use_device(X,Y)` directive, which ensures that the kernels see the address of the ACC data on the device (kernels are launched from host, so they need device pointers to access device data).

The cuBLAS SAXPY routine follows a similar idea, and its used to illustrate how CUDA libraries, or any other GPU-based library, can also operate on ACC data.

## Building and running

The provided CMake infrastructure ensures appropriate compilation:

```bash
mkdir build
cd build
cmake ..
make
```

Ensure that the NVHPC compiler is available in your environment, as well as the CUDA toolkit.