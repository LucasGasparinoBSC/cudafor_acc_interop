module myKernels
    use cudafor
    implicit none
    contains
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
  use cublas
  use myKernels
  integer, parameter :: N = 2**20
  integer:: blockSize, gridSize
  real, allocatable, dimension(:) :: X, Y

  blockSize = 256
  gridSize = (N + blockSize - 1) / blockSize

  allocate(X(N), Y(N))
  !$acc enter data create(x,y)
  !$acc kernels
  X(:) = 3.0
  Y(:) = 5.0
  !$acc end kernels

  !$acc host_data use_device(x,y)
  call myKernel<<<gridSize, blockSize>>>(Y, N)
  call myKernel2<<<gridSize, blockSize>>>(X, Y, N)
  call cublassaxpy(N, 2.0, x, 1, y, 1)
  !$acc end host_data
  !$acc update self(y)

  print *, y(1)

  !$acc host_data use_device(y)
  call myKernel<<<gridSize, blockSize>>>(Y, N)
  !$acc end host_data
    !$acc update self(y)

    print *, y(1)
end program