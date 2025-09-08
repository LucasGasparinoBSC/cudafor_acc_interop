! Module containing the host and device objects
module myTypes
    implicit none
        ! Host object
        type :: myType_h
            real(4), allocatable :: x(:), y(:)
        end type myType_h

        ! Device object
        ! Note: the object itself is on the host, but its components are on the device
        type :: myType_d
            real(4), device, allocatable :: x(:), y(:)
        end type myType_d
end module myTypes

module myKernels
    use openacc
    use myTypes
    implicit none
    contains
        subroutine addAttributes(obj, N)
            implicit none
            integer, value :: N
            type(myType_d), intent(inout) :: obj
            integer :: i

            !$acc parallel loop
            do i = 1, N
                obj%y(i) = obj%x(i) + obj%y(i)
            end do
            !$acc end parallel loop
        end subroutine addAttributes

        subroutine attributeDot(obj, N, result)
            implicit none
            integer, value :: N
            type(myType_d), intent(in) :: obj
            real(8), intent(out) :: result
            integer :: i

            result = 0.0
            !$acc parallel loop reduction(+:result)
            do i = 1, N
                result = result + real(obj%x(i) * obj%y(i),8)
            end do
            !$acc end parallel loop
        end subroutine attributeDot

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
    ! Note: only the attributes are allocated. For the device object, the object itself is on the host,
    ! but its components are on the device, so allocate will only create device attributes.
    allocate(h_obj%x(N), h_obj%y(N))
    allocate(d_obj%x(N), d_obj%y(N))

    ! Initialize device object with ACC PARALLEL LOOP
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

    ! Driver dot product ACC kernel
    dotResult = 0.0
    !$acc parallel loop reduction(+:dotResult)
    do i = 1, N
        dotResult = dotResult + real(d_obj%x(i) * d_obj%y(i),8)
    end do
    !$acc end parallel loop

    print*, "dotResult = ", dotResult

    ! Call the kernel addAttributes kernel
    call addAttributes(d_obj, N)

    ! Call the attributeDot kernel
    call attributeDot(d_obj, N, dotResult)
    print*, "After addAttributes, dotResult = ", dotResult

    ! Print y(1)
    h_obj%y = d_obj%y
    print*, "After addAttributes, h_obj%y(1) = ", h_obj%y(1)
end program main