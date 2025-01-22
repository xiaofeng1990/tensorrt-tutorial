// cuda runtime 头文件
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <chrono>

#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{

    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);

        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

void launch();

int main()
{

    cudaDeviceProp prop;
    checkRuntime(cudaGetDeviceProperties(&prop, 0));
    printf("prop.sharedMemPerBlock = %.2f KB\n", prop.sharedMemPerBlock / 1024.0f);

    printf("prop.maxGridSize x = %d, y = %d z = %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("prop.maxThreadsDim x = %d, y = %d z = %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("prop.maxThreadsPerBlock %d \n", prop.maxThreadsPerBlock);
    launch();
    checkRuntime(cudaPeekAtLastError());
    // 等待device运行完毕
    checkRuntime(cudaDeviceSynchronize());
    printf("done\n");
    return 0;
}