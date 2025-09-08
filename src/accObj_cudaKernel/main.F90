! Simple DDT with allocatable arrays
module myTypes
    implicit none
        type :: myType_1
            real(4), allocatable :: x(:), y(:)
        end type myType_1
end module myTypes

! CUDA Fortran kernels
module myKernels
    use myTypes
    use cudafor
    implicit none
    contains
        ! Vector addition using object attributes
        attributes(global) subroutine myKernel_1(obj,n)
            integer :: gid
            integer, value :: n
            type(myType_1), intent(inout) :: obj ! Object has to contain device pointers

            gid = threadIdx%x + (blockIdx%x - 1) * blockDim%x
            if (gid <= n) then
                obj%y(gid) = obj%x(gid) + obj%y(gid)
            end if
        end subroutine myKernel_1
end module myKernels

program main
    use myTypes
    use myKernels
    use cudafor
    use openacc

    implicit none

    integer :: nblocks, nthreads, i
    integer, parameter :: N = 512*512
    type(myType_1) :: obj_test

    ! Create a host/device object, ensuring attributes are set correctly
    allocate(obj_test%x(N), obj_test%y(N))
    ! Both the object and its attributes must be created on the device
    !$acc enter data create(obj_test, obj_test%x, obj_test%y)

    ! ACC KERNELS initialization ensures data exists on device before use
    !$acc kernels
    obj_test%x(:) = 3.0
    obj_test%y(:) = 5.0
    !$acc end kernels

    ! Update host and print (only update attributes)
    !$acc update host(obj_test%x, obj_test%y)
    print*, obj_test%x(1), obj_test%y(1)

    ! Kernel launch parameters
    nthreads = 256
    nblocks = (N + nthreads - 1) / nthreads

    ! Launch kernel (using host_data to pass device pointer)
    ! NOTE: here the pointer is to the object, NOT to its attributes
    !$acc host_data use_device(obj_test)
    call myKernel_1<<<nblocks, nthreads>>>(obj_test, N)
    !$acc end host_data

    ! Update host attributes and print
    !$acc update self(obj_test%x, obj_test%y)
    print*, obj_test%x(1), obj_test%y(1)
end program main