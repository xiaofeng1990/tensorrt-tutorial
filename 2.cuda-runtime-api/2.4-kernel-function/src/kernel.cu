#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CPU调用，在GPU上执行
__global__ void test_print_kernel(const float *pdata, int ndata)
{
    // threadIdx blockIdx  blockDim 内置变量

    // threadIdx;
    // blockIdx;
    // blockDim; threadIdx 表示第几个block
    // gridDim; blockIdx 表示第几个grid

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    /*
           dims           indexs
        gridDim.z  1     blockIdx.z  0
        gridDim.y  1     blockIdx.y  0
        gridDim.x  2     blockIdx.x  0-1
        blockDim.z 1     threadIdx.z 0
        blockDim.y 1     threadIdx.y 0
        blockDim.x 10    threadIdx.x 0-9

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    printf("Element[%d] = %f, threadIdx.x = %d, blockIdx.x=%d, blockDim.x=%d\n", idx, pdata[idx], threadIdx.x, blockIdx.x, blockDim.x);
}

void test_print(const float *pdata, int ndata)
{
    dim3 gridDim;
    dim3 blockDim;
    int nthreads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;

    // gridDim(21亿, 65536, 65536) // 可以通过runtime API 查询
    // blockDim(1024, 64, 64)   blockDim.x * blockDim.y * blockDim.z<=1024
    //  <<< gridDim, blockDim,  bytes_of_shared_memory, stream>>>
    test_print_kernel<<<dim3(2), dim3(ndata / 2), 0, nullptr>>>(pdata, ndata);
    // 在核函数执行结束后，通过 cudaPeekAtLastError 判断是否执行错误
    //  cudaPeekAtLastError 和 cudaGetLastError 都可以获取错误代码
    //  cudaGetLastError 是获取错误代码后并清楚，也就是在执行一次 cudaGetLastError 获取到的会是 success
    //  而 cudaPeekAtLastError 是获取当前错误，但是再次执行 cudaPeekAtLastError 或者 cudaGetLastError 拿到的还是那个错
    //  cuda 的错误会传递，如果这里出错，不移除，那么后续的任意api的返回值都会是这个错误，都会失败

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}
