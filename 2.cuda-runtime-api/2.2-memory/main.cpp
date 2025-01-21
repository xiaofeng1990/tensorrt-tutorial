// cuda驱动头文件
#include <cuda.h>
// cuda runtime 头文件
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>

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

int main()
{
    int device_id = 0;
    checkRuntime(cudaSetDevice(device_id));

    // global memory on device(GPU)
    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));
    printf("memory_device = %p\n", memory_device);

    // pageable memory on host(CPU) GPU 不可以字节访问
    float *memory_host = new float[100];
    memory_host[2] = 520.25;
    checkRuntime(cudaMemcpy(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice));
    printf("memory_device = %p\n", memory_device);

    // pinned memory page locked memory GPU 可以字节访问
    float *memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    printf("memory_page_locked = %p\n", memory_page_locked);
    // 同步copy
    checkRuntime(cudaMemcpy(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost));

    printf("memory_page_locked[2] = %f\n", memory_page_locked[2]);

    checkRuntime(cudaFreeHost(memory_page_locked));
    delete[] memory_host;
    checkRuntime(cudaFree(memory_device));

    return 0;
}