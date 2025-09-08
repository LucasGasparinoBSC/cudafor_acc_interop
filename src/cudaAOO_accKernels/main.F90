module myTypes
    use cudafor
    implicit none
        ! Host object
        type :: myType_h
            real(4), pointer :: hx(:) => null()
            real(4), pointer :: hy(:) => null()
        end type myType_h

        ! Device object
        type :: myType_d
            real(4), device, allocatable :: dx(:)
            real(4), device, allocatable :: dy(:)
        end type myType_d
end module myTypes

module myKernels
    use myTypes
    use openacc
    implicit none
    contains
        subroutine vecAdd(obj, nobj, ndata)
            implicit none
            integer, value :: nobj, ndata
            type(myType_d) :: obj(nobj)
            integer :: iobj, idata

            !$acc parallel loop gang
            do iobj = 1, nobj
                !$acc loop vector
                do idata = 1, ndata
                    obj(iobj)%dx(idata) = obj(iobj)%dx(idata) + obj(iobj)%dy(idata)
                end do
            end do
            !$acc end parallel loop
        end subroutine vecAdd
end module myKernels

program main
    use myTypes
    use myKernels
    use cudacInterface
    use cudafor
    use openacc

    implicit none

    integer :: iobj, idata
    integer, parameter :: nObj = 1000
    integer, parameter :: N = 512
    real(8) :: dotResult
    type(myType_h), allocatable :: aoo_h(:)
    type(myType_d), allocatable :: aoo_h2d(:)

    ! Allocate the host AOO
    allocate(aoo_h(nObj))
    do iobj = 1, nObj
        allocate(aoo_h(iobj)%hx(N))
        allocate(aoo_h(iobj)%hy(N))
        aoo_h(iobj)%hx = 0.0
        aoo_h(iobj)%hy = 0.0
    end do

    ! Allocate the host-to-device AOO (host descriptor, device attributes)
    allocate(aoo_h2d(nObj)) ! Host memory
    do iobj = 1, nObj
        aoo_h2d(iobj)%dx = aoo_h(iobj)%hx ! Device memory
        aoo_h2d(iobj)%dy = aoo_h(iobj)%hy ! Device memory
    end do

    !$acc parallel loop gang
    do iobj = 1, nObj
        !$acc loop vector
        do idata = 1, N
            aoo_h2d(iobj)%dx(idata) = 3.0
            aoo_h2d(iobj)%dy(idata) = 5.0
        end do
    end do
    !$acc end parallel loop

    ! Copy back and print 1st element of each array
    do iobj = 1, nObj
        aoo_h(iobj)%hx = aoo_h2d(iobj)%dx
        aoo_h(iobj)%hy = aoo_h2d(iobj)%dy
    end do
    print *, 'aoo_h(1)%hx(1) = ', aoo_h(1)%hx(1)
    print *, 'aoo_h(nObj)%hx(1) = ', aoo_h(nObj)%hx(1)
    print *, 'aoo_h(1)%hy(1) = ', aoo_h(1)%hy(1)
    print *, 'aoo_h(nObj)%hy(1) = ', aoo_h(nObj)%hy(1)

    ! call the vecAdd kernel
    call vecAdd(aoo_h2d, nObj, N)

    ! Copy back and print 1st element of each array
    do iobj = 1, nObj
        aoo_h(iobj)%hx = aoo_h2d(iobj)%dx
        aoo_h(iobj)%hy = aoo_h2d(iobj)%dy
    end do
    print *, 'After vecAdd:'
    print *, 'aoo_h(1)%hx(1) = ', aoo_h(1)%hx(1)
    print *, 'aoo_h(nObj)%hx(1) = ', aoo_h(nObj)%hx(1)
    print *, 'aoo_h(1)%hy(1) = ', aoo_h(1)%hy(1)
    print *, 'aoo_h(nObj)%hy(1) = ', aoo_h(nObj)%hy(1)

end program main