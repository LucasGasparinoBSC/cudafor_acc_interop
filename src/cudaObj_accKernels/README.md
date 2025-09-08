# CUDA objects on ACC kernels

In this example, CUDA Fortran is used to create a device object that is passed to a variety of ACC kernels.

## Details

In this example, the driver program uses CUDA Fortran to create a device object that is later passed to OpenACC kernels. The object itself lives on host, but its attributes are classified as `device`, meaning that the data members of the object are allocated on device memory when ALLOCATE is called on them. A host object is also created, separatedly, to allow for printing of attributes.

Once the device copy of the object exists, it is a simple matter to use it on the OpenACC kernels; no special ACC directives are required. The object itself is never updated, only its attributes. The driver launches both an internal kernel (using the `acc parallel loop` directive) and 2 external kernels contained in the `myKernels` module. In this case, since the object is being passed to the subroutines, theres no need to declare the attributes with `device` qualification, simplifying the routine signature.

## Building and running

The provided CMake infrastructure ensures appropriate compilation:

```bash
mkdir build
cd build
cmake ..
make
```

Ensure that the NVHPC compiler is available in your environment, as well as the CUDA toolkit.