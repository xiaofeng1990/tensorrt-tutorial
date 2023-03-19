#include <cuda.h>
#include <stdio.h>
#include <string.h>

// #define checkDriver(op)                                                                                                \
//     do                                                                                                                 \
//     {                                                                                                                  \
//         auto code = (op);                                                                                              \
//         if (code != CUresult::CUDA_SUCCESS)                                                                            \
//         {                                                                                                              \
//             const char *err_name = nullptr;                                                                            \
//             const char *err_message = nullptr;                                                                         \
//             cuGetErrorName(code, &err_name);                                                                           \
//             cuGetErrorString(code, &err_message);                                                                      \
//             printf("%s:%d  %s failed. \n  code = %s, message = %s\n", __FILE__, __LINE__, #op, err_name, err_message); \
//             return -1;                                                                                                 \
//         }                                                                                                              \
//     } while (0)

#define checkDriver(op) __check_cuda_driver((op), #op, __FILE__, __LINE__)

bool __check_cuda_driver(CUresult code, const char *op, const char *file, int line)
{

    if (code != CUresult::CUDA_SUCCESS)
    {
        const char *err_name = nullptr;
        const char *err_message = nullptr;
        cuGetErrorName(code, &err_name);
        cuGetErrorString(code, &err_message);
        printf("%s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

int main()
{
    if (!checkDriver(cuInit(0)))
    {
        return -1;
    }

    // 获取 cuda 驱动版本 比如10.2
    int driver_version = 0;
    if (!checkDriver(cuDriverGetVersion(&driver_version)))
    {
        return -1;
    }
    printf("CUDA Driver version is %d\n", driver_version);

    // 获取当前设备信息
    char device_name[100];
    CUdevice device = 0;
    if (!checkDriver(cuDeviceGetName(device_name, sizeof(device_name), device)))
    {
        return -1;
    }
    printf("Device %d name is %s\n", device, device_name);
    return 0;
}