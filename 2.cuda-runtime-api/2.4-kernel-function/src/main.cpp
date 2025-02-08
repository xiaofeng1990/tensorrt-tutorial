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

void test_print(const float *pdata, int ndata);

int main()
{
    float *parray_host = nullptr;
    float *parray_device = nullptr;
    int narray = 32;
    int array_bytes = sizeof(float) * narray;
    printf("array_bytes  = %d \n", array_bytes);
    // host memory
    parray_host = new float[narray];

    checkRuntime(cudaMalloc(&parray_device, array_bytes));

    for (int i = 0; i < narray; ++i)
    {
        parray_host[i] = i;
    }
    // 从host同步复制数据到device
    checkRuntime(cudaMemcpy(parray_device, parray_host, array_bytes, cudaMemcpyHostToDevice));
    // 使用kernel函数加速打印device memory
    test_print(parray_device, narray);
    checkRuntime(cudaDeviceSynchronize());

    // free host and device memory
    checkRuntime(cudaFree(parray_device));
    delete[] parray_host;

    return 0;
}