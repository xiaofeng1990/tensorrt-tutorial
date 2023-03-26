#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <device_launch_parameters.h>

__global__ void sum_kernel(float *array, int n, float *output)
{
    int position = blockIdx.x * blockDim.x + threadIdx.x;
    // 使用 extern声明外部的动态大小共享内存，由启动核函数的第三个参数指定
    // 这个cache 的大小为 block_size * sizeof(float)
    extern __shared__ float cache[];
    int block_size = blockDim.x;
    int lane = threadIdx.x;
    float value = 0;

    printf("thread position = %d, threadIdx.x = %d, blockIdx.x=%d, blockDim.x=%d\n", position, threadIdx.x, blockIdx.x, blockDim.x);

    if (position < n)
    {

        value = array[position];
        printf("position = %d, value = %f, n = %d \n", position, value, n);
    }
    for (int i = block_size / 2; i > 0; i /= 2)
    {
        // 等待block内的所有线程储存完毕
        cache[lane] = value;
        __syncthreads();
        if (lane < i)
        {
            value += cache[lane + i];
        }
        // 等待block内的所有线程读取完毕
        __syncthreads();
    }
    // 最后所有和都归于每个block中的thread0
    if (lane == 0)
    {
        printf("block %d value = %f\n", blockIdx.x, value);
        // 由于可能动用了多个block，所以汇总结果的时候需要用atomicAdd。（注意这里的value仅仅是一个block的threads reduce sum 后的结果）
        // 将每个block中的结果加到output上
        atomicAdd(output, value);
    }
}

void launch_reduce_sum(float *array, int n, float *output)
{
    const int nthreads = 512;
    int block_size = n < nthreads ? n : nthreads;
    int grid_size = (n + block_size - 1) / block_size;
    // 这里要求 block_size 必须是 2 的幂次
    float block_sqrt = log2(block_size);
    printf("old block_size = %d, block_sqrt = %.2f\n", block_size, block_sqrt);
    // 向上取整
    block_sqrt = ceil(block_sqrt);
    printf("block_sqrt = %.2f\n", block_sqrt);
    block_size = pow(2, block_sqrt);
    printf("block_size = %d, grid_size = %d\n", block_size, grid_size);

    sum_kernel<<<grid_size, block_size, block_size * sizeof(float), nullptr>>>(array, n, output);
}