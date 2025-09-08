module myTypes
    implicit none
        ! Host object
        type :: myType_h
            real(4), allocatable :: x(:), y(:)
        end type myType_h

        ! Device object
        type :: myType_d
            real(4), device, allocatable :: x(:), y(:)
        end type myType_d
end module myTypes

module myKernels
    use myTypes
    use cudafor
    implicit none
    contains
end module myKernels

program main
    use myTypes
    use myKernels
    use cudafor
    use openacc

    implicit none

    integer :: i
    integer, parameter :: N = 512
    real(8) :: dotResult

    ! Host version of the object
    type(myType_h) :: h_obj

    ! Device version of the object
    type(myType_d) :: d_obj

    ! Allocate obj memory
    allocate(h_obj%x(N), h_obj%y(N))
    allocate(d_obj%x(N), d_obj%y(N))

    !$acc parallel loop
    do i = 1, N
        d_obj%x(i) = 3.0 + real(i,4)
        d_obj%y(i) = 5.0 + real(i,4)
    end do
    !$acc end parallel loop

    ! Copy to host
    h_obj%x = d_obj%x
    h_obj%y = d_obj%y
    print*, "h_obj%x(1) = ", h_obj%x(1), " h_obj%y(1) = ", h_obj%y(1)

    dotResult = 0.0
    !$acc parallel loop reduction(+:dotResult)
    do i = 1, N
        dotResult = dotResult + real(d_obj%x(i) * d_obj%y(i),8)
    end do
    !$acc end parallel loop

    print*, "dotResult = ", dotResult
end program main