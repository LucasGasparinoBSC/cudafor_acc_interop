! Module containing user-defined CUDA Fortran kernels
module myKernels
    ! CUDAFOR module enables CUDA Fortran features
    use cudafor
    implicit none
    contains
        ! Adds 1 to each element of Y
        attributes(global) subroutine myKernel(Y, N)
            implicit none
            integer, value :: N
            real, intent(inout) :: Y(N)
            integer :: gid

            gid = threadIdx%x + (blockIdx%x - 1) * blockDim%x
            if (gid <= N) then
                Y(gid) = Y(gid) + 1.0
            end if
        end subroutine myKernel

        ! Simple vector addition
        attributes(global) subroutine myKernel2(X, Y, N)
            implicit none
            integer, value :: N
            real, intent(in) :: X(N)
            real, intent(inout) :: Y(N)
            integer :: gid

            gid = threadIdx%x + (blockIdx%x - 1) * blockDim%x
            if (gid <= N) then
                Y(gid) = Y(gid) + X(gid)
            end if
        end subroutine myKernel2
end module myKernels

program main
    use cublas ! enables use of cuBLAS library
    use myKernels
    integer, parameter :: N = 2**20
    integer:: blockSize, gridSize
    real, allocatable, dimension(:) :: X, Y

    ! Define the kernel launch parameters
    blockSize = 256
    gridSize = (N + blockSize - 1) / blockSize

    ! Create the data on both host and device
    ! NOTE: all GPU kernels will work with this data
    allocate(X(N), Y(N))
    !$acc enter data create(x,y)

    ! Initializing with ACC KERNELS ensures the arrays exist on device memory
    !$acc kernels
    X(:) = 3.0
    Y(:) = 5.0
    !$acc end kernels

    ! Launch th kernels and perform cuBLAS SAXPY operation
    ! NOTE: use host_data to pass device pointers to kernels and cuBLAS
    !$acc host_data use_device(x,y)
    call myKernel<<<gridSize, blockSize>>>(Y, N)
    call myKernel2<<<gridSize, blockSize>>>(X, Y, N)
    call cublassaxpy(N, 2.0, x, 1, y, 1)
    !$acc end host_data

    ! Update host before printing
    !$acc update self(y)

    print *, y(1)

    ! New kernel launch to ensure data still exists on device
    !$acc host_data use_device(y)
    call myKernel<<<gridSize, blockSize>>>(Y, N)
    !$acc end host_data
    !$acc update self(y)

    print *, y(1)
end program