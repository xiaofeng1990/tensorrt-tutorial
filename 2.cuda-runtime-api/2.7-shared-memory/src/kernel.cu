#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//////////////////////demo1 //////////////////////////
/*
demo1 主要为了展示查看静态和动态共享变量的地址
 */
// 静态共享内存
const size_t static_shared_memory_num_element = 6 * 1024; // 6KB
__shared__ char static_shared_memory[static_shared_memory_num_element];
__shared__ char static_shared_memory2[2];

__global__ void demo1_kernel()
{
    // 动态共享内存
    extern __shared__ char dynamic_shared_memory[]; // 静态共享变量和动态共享变量在kernel函数内/外定义都行，没有限制
    extern __shared__ char dynamic_shared_memory2[];
    printf("static_shared_memory = %p\n", static_shared_memory); // 静态共享变量，定义几个地址随之叠加
    printf("static_shared_memory2 = %p\n", static_shared_memory2);
    printf("dynamic_shared_memory = %p\n", dynamic_shared_memory); // 动态共享变量，无论定义多少个，地址都一样
    printf("dynamic_shared_memory2 = %p\n", dynamic_shared_memory2);

    if (blockIdx.x == 0 && threadIdx.x == 0) // 第一个thread
        printf("Run kernel.\n");
}

/////////////////////demo2//////////////////////////////////
/*
demo2 主要是为了演示的是如何给 共享变量进行赋值
 */

// 定义共享变量，但是不能给初始值，必须由线程或者其他方式赋值

__global__ void demo2_kernel()
{
    __shared__ int shared_value1;
    __shared__ int shared_value2;

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

    // 在每一个block中的threadIdx.x 为0的时候赋值，其他线程使用共享内存
    if (threadIdx.x == 0)
    {
        // block 0 全部是123，64
        if (blockIdx.x == 0)
        {
            shared_value1 = 123;
            shared_value2 = 64;
        }
        else
        {
            // block 1 全部是 456，89
            shared_value1 = 456;
            shared_value2 = 89;
        }
    }

    // block中其它阻塞到这里, 等待所有线程到达
    __syncthreads();
    printf("%d.%d. shared_value1 = %d[%p], shared_value2 = %d[%p]\n",
           blockIdx.x, threadIdx.x,
           shared_value1, &shared_value1,
           shared_value2, &shared_value2);
}

void launch()
{
    demo1_kernel<<<1, 1, 12, nullptr>>>();
    demo2_kernel<<<2, 5, 0, nullptr>>>();
}