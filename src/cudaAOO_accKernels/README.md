# CUDA AOO on ACC kernels

In this example, CUDA Fortran is used to create an array of objects (AOO) that is passed to a variety of ACC kernels.

## Details

This is a more interesting example than the previous ones: CUDA cannot allocate the AOO on device, then allocate its attributes as well! The solution is surprisingly simple: create the AOO on host, its attributes on device, then launch the kernels without including a PRESENT statement for the AOO itself.

Even in the case that the object is passed to a routine, which then executes the ACC kernel, this solution is completely valid.

## Building and running

The provided CMake infrastructure ensures appropriate compilation:

```bash
mkdir build
cd build
cmake ..
make
```

Ensure that the NVHPC compiler is available in your environment, as well as the CUDA toolkit.