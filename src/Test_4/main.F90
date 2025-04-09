module myTypes
    implicit none
        type :: myType_1
            real(4), allocatable :: x(:), y(:)
        end type myType_1
end module myTypes

module myKernels
    use myTypes
    use cudafor
    implicit none
    contains
        attributes(global) subroutine myKernel_1(objArr,nobj,n)
            implicit none
            integer :: objId, arrId
            integer, value :: nobj, n
            type(myType_1), intent(inout) :: objArr(nobj)

            objId = blockIdx%x
            arrId = threadIdx%x

            objArr(objId)%y(arrId) = objArr(objId)%x(arrId) + objArr(objId)%y(arrId)
        end subroutine myKernel_1
end module myKernels

program main
    use myTypes
    use myKernels
    use cudafor
    use openacc

    implicit none

    integer :: nblocks, nthreads, iObj, i, irun
    integer, parameter :: N = 256
    integer, parameter :: numObj = 1000
    type(myType_1), allocatable :: obj_array(:)

    allocate(obj_array(numObj))
    !$acc enter data create(obj_array)
    do i = 1, numObj
        allocate(obj_array(i)%x(N), obj_array(i)%y(N))
        !$acc enter data create(obj_array(i), obj_array(i)%x, obj_array(i)%y)
    end do

    !$acc parallel loop gang
    do iObj = 1, numObj
        !$acc loop vector
        do i = 1, N
            obj_array(iObj)%x(i) = 3.0 + real(iObj, 4)
            obj_array(iObj)%y(i) = 5.0 + real(iObj, 4)
        end do
    end do
    !$acc end parallel loop

    do iObj = 1, numObj
        !$acc update host(obj_array(iObj)%x, obj_array(iObj)%y)
        print*, "obj_array(", iObj, ") x(1) = ", obj_array(iObj)%x(1), " y(1) = ", obj_array(iObj)%y(1)
    end do

    nthreads = N
    nblocks = numObj
    !$acc host_data use_device(obj_array)
    do irun = 1,10
        call myKernel_1<<<nblocks, nthreads>>>(obj_array, numObj, N)
    end do
    !$acc end host_data

    do iObj = 1, numObj
        !$acc update host(obj_array(iObj)%y)
        print*, "obj_array(", iObj, ")", " y(1) = ", obj_array(iObj)%y(1)
    end do

end program main