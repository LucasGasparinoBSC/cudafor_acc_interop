program main
    use cudafor
    use openacc

    implicit none
    integer :: i
    integer, parameter :: N = 2**20
    real, device, allocatable :: X(:), Y(:)
    real, allocatable :: Y_host(:)
    real(8) :: dot_result

    allocate(Y_host(N))
    allocate(X(N), Y(N))
    !$acc kernels
    X(:) = 3.0
    Y(:) = 5.0
    !$acc end kernels

    !$acc parallel loop
    do i = 1, N
        Y(i) = Y(i) + X(i)
    end do
    !$acc end parallel loop

    Y_host = Y
    print *, Y_host(1)

    dot_result = 0.0d0
    !$acc parallel loop reduction(+:dot_result)
    do i = 1, N
        dot_result = dot_result + real(Y(i)*X(i),8)
    end do
    !$acc end parallel loop

    print *, "Dot product result: ", dot_result
end program main