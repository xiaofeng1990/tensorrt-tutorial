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

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));

    float *memory_device = nullptr;
    checkRuntime(cudaMalloc(&memory_device, 100 * sizeof(float)));
    printf("memory_device = %p\n", memory_device);

    float *memory_host = new float[100];
    memory_host[2] = 520.25;
    checkRuntime(cudaMemcpyAsync(memory_device, memory_host, sizeof(float) * 100, cudaMemcpyHostToDevice, stream));
    printf("memory_device = %p\n", memory_device);

    float *memory_page_locked = nullptr;
    checkRuntime(cudaMallocHost(&memory_page_locked, 100 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(memory_page_locked, memory_device, sizeof(float) * 100, cudaMemcpyDeviceToHost, stream));
    printf("memory_page_locked[2] = %f\n", memory_page_locked[2]);
    // 等待stream中的队列处理完
    checkRuntime(cudaStreamSynchronize(stream));

    printf("memory_page_locked[2] = %f\n", memory_page_locked[2]);

    checkRuntime(cudaFreeHost(memory_page_locked));
    checkRuntime(cudaFree(memory_device));
    checkRuntime(cudaStreamDestroy(stream));
    delete[] memory_host;
    return 0;
}