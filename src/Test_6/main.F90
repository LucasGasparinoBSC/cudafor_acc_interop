module myTypes
    use cudafor
    implicit none
        ! Host object
        type :: myType_h
            real(4), allocatable :: x(:), y(:)
        end type myType_h

        ! Device object
        type :: myType_d
            real(4), device, pointer :: dx(:), dy(:)
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

    integer :: iobj, i, ierr
    integer, parameter :: nObj = 1000
    integer, parameter :: N = 512
    real(8) :: dotResult

    type(myType_d), allocatable, device :: obj_array_d(:)

    allocate(obj_array_d(nObj))
    do iobj = 1, nObj
        allocate(obj_array_d(iobj)%dx(N), obj_array_d(iobj)%dy(N))
    end do
end program main