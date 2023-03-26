#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void vector_add_kernel(const float *a, const float *b, float *c, int ndata)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= ndata)
        return;

    /*    dims                 indexs
        gridDim.z            blockIdx.z
        gridDim.y            blockIdx.y
        gridDim.x            blockIdx.x
        blockDim.z           threadIdx.z
        blockDim.y           threadIdx.y
        blockDim.x           threadIdx.x

        Pseudo code:
        position = 0
        for i in 6:
            position *= dims[i]
            position += indexs[i]
    */
    c[idx] = a[idx] + b[idx];
}

void vector_add(const float *a, const float *b, float *c, int ndata)
{
    // nthreads的取值，不能大于block能取值的最大值。一般可以直接给512、256，性能就是比较不错的
    const int nthreads = 1024;
    //  如果ndata < nthreads 那block_size = ndata就够了
    int block_size = ndata < nthreads ? ndata : nthreads;
    // 其含义是我需要多少个blocks可以处理完所有的任务
    int grid_size = (ndata + block_size - 1) / block_size;
    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);

    vector_add_kernel<<<grid_size, block_size, 0, nullptr>>>(a, b, c, ndata);

    cudaError_t code = cudaPeekAtLastError();
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("kernel error %s:%d  test_print_kernel failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, err_name, err_message);
    }
}