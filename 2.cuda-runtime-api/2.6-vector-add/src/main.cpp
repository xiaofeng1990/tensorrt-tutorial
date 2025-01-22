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

void vector_add_host(const float *a, const float *b, float *c, int ndata)
{
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    for (int i = 0; i < ndata; i++)
    {
        c[i] = a[i] + b[i];
    }
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());

    printf("host vector add [0--%d] time %ld ms\n", ndata, timestamp2 - timestamp1);
}

void vector_add(const float *a, const float *b, float *c, int ndata);
// using namespace std::chrono;

int main()
{

    const int size = 100000000;
    // const int size = 1028;
    // pageable memory
    float *vector_a = new float[size];
    float *vector_b = new float[size];
    float *vector_c = new float[size];

    // GPU 访问 pinned memory page locked memory 更快
    // float *vector_a;
    // float *vector_b;
    // float *vector_c;
    // checkRuntime(cudaMallocHost(&vector_a, size * sizeof(float)));
    // checkRuntime(cudaMallocHost(&vector_b, size * sizeof(float)));
    // checkRuntime(cudaMallocHost(&vector_c, size * sizeof(float)));

    for (int i = 0; i < size; i++)
    {

        vector_a[i] = (float)i;
        vector_b[i] = static_cast<float>(i);
    }
    vector_add_host(vector_a, vector_b, vector_c, size);

    float *vector_a_device = nullptr;
    float *vector_b_device = nullptr;
    float *vector_c_device = nullptr;

    checkRuntime(cudaMalloc(&vector_a_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_b_device, size * sizeof(float)));
    checkRuntime(cudaMalloc(&vector_c_device, size * sizeof(float)));
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    checkRuntime(cudaMemcpy(vector_a_device, vector_a, size * sizeof(float), cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(vector_b_device, vector_b, size * sizeof(float), cudaMemcpyHostToDevice));

    vector_add(vector_a_device, vector_b_device, vector_c_device, size);

    checkRuntime(cudaMemcpy(vector_c, vector_c_device, size * sizeof(float), cudaMemcpyDeviceToHost));
    checkRuntime(cudaDeviceSynchronize()); // 进行同步，这句话以上的代码全部可以异步操作

    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("cuda vector add [0--%d] time %ld ms\n", size, timestamp2 - timestamp1);

    // for (int i = 0; i < size; ++i)
    // {
    //     printf("vector_c[%d] = %f\n", i, vector_c[i]);
    // }
    checkRuntime(cudaFree(vector_a_device));
    checkRuntime(cudaFree(vector_b_device));
    checkRuntime(cudaFree(vector_c_device));

    // checkRuntime(cudaFreeHost(vector_a));
    // checkRuntime(cudaFreeHost(vector_b));
    // checkRuntime(cudaFreeHost(vector_c));

    delete vector_a;
    delete vector_b;
    delete vector_c;
    return 0;
}