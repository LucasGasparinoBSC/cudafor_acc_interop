# ACC DDT on CUDA kernel

In this example, OpenACC is used to create a simple object with allocatable attributes, which is then passed to a CUDA kernel.

## Details

This example is the starting point for performing interoperability between OpenACC and CUDA Fortran using object orientation: the driver program creates a single object on both host and device, then passes it to a CUDA kernel. The object attributes are allocatable arrays, which is what the CUDA kernel will operate on.

From the data management side, the `acc enter data create` directive is used to create both the object AND its attributes, right after the host allocates the attributes. The rest of the code is similar to the `acc2cuda` example: before the kernel call, the `acc host_data use_device` directive is used to get the device pointer of the object, which is then passed to the kernel. Notice that only the OBJECT needs to be passed to the kernel, since the attributes are part of the object.

The final part involves updating the host copies of the attributes before printing: no need to update the object as well.

## Building and running

The provided CMake infrastructure ensures appropriate compilation:

```bash
mkdir build
cd build
cmake ..
make
```

Ensure that the NVHPC compiler is available in your environment, as well as the CUDA toolkit.