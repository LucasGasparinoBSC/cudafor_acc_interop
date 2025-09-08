! Simple module containing an OpenACC kernel for dot product
module mod_acc_kernels
    implicit none
    contains
        subroutine acc_dotproduct(N, X, Y, result)
            integer, value :: N
            real, device, intent(in) :: X(N), Y(N) ! Notice device attribute to match caller
            real(8), intent(out) :: result
            integer :: i
            result = 0.0d0
            !$acc parallel loop reduction(+:result)
            do i = 1, N
                result = result + real(Y(i)*X(i),8)
            end do
            !$acc end parallel loop
        end subroutine acc_dotproduct
end module mod_acc_kernels

! Creates device data with CUDA Fortran and uses OpenACC to perform operations
program main
    use cudafor
    use openacc
    use mod_acc_kernels

    implicit none
    integer :: i
    integer, parameter :: N = 2**20
    real, allocatable :: Y_host(:)
    real(8) :: dot_result

    ! The device attribute (CUDAFOR exclusive) ensures the allocate command creates device memory
    real, device, allocatable :: X(:), Y(:)

    ! Simple host allocation
    allocate(Y_host(N))

    ! This is equivalent to a cudaMalloc operation
    allocate(X(N), Y(N))

    ! ACC KERNELS operates directly on device data, no further directives needed
    !$acc kernels
    X(:) = 3.0
    Y(:) = 5.0
    !$acc end kernels

    ! Driver can easily launch the kernel on device data
    !$acc parallel loop
    do i = 1, N
        Y(i) = Y(i) + X(i)
    end do
    !$acc end parallel loop

    ! Copy results back to host, essentually a cudaMemcpy operation
    Y_host = Y
    print *, Y_host(1)

    ! Call the OpenACC dot product kernel, using device data
    ! NOTE: the dummy argument declarations in the subroutine must match
    call acc_dotproduct(N, X, Y, dot_result)

    ! No need to copy a scalar result back to host, it's done automatically
    print *, "Dot product result: ", dot_result
end program main