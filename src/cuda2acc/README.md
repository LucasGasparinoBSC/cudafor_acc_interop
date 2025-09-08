# CUDA data on ACC kernels

In this example, CUDA Fortran is used to create device data that is later accessed by OpenACC kernels.

## Details

In this example, the driver program uses CUDA Fortran data management to ensure device data exists for the OpenACC kernels to operate on. In essence, allocatable arrays are declared with the `device` attribute, ensuring that calls to `allocate` and `deallocate` are actually calls to `cudaMalloc` and `cudaFree`; as well, the assignment operator is overloaded to perform `cudaMemcpy` operations in either direction, depending on the attributes of the source and destination arrays.

Later on, OpenACC kernels are launched to operate on that data. In this case, no special directives are needed to ensure the kernels see the device data, as the arrays are already device arrays. The main trick here is the argument definition of the `acc_dotproduct` subroutine:

```fortran
subroutine acc_dotproduct(N, X, Y, result)
    integer, value :: N
    real, device, intent(in) :: X(N), Y(N) ! Notice device attribute to match caller
    real(8), intent(out) :: result
```

Passing N by value ensures no implicit copying of the scalar value is attempted, so it does not need to be created beforehand on the driver code, which is typical of CUDA and mimics the behaviour of CUDA C usage. As for the arrays, the `device` attribute is required to match the caller, otherwise the compiler will fail with an error indicating a mismatch in attributes.

## Building and running

The provided CMake infrastructure ensures appropriate compilation:

```bash
mkdir build
cd build
cmake ..
make
```

Ensure that the NVHPC compiler is available in your environment, as well as the CUDA toolkit.