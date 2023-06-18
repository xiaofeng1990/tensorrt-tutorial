#include "cuda_tools.hpp"
#include <stdio.h>
#include <stdarg.h>
#include <string>
#include "logger.hpp"

using namespace std;

namespace CUDATools
{
    bool check_runtime(cudaError_t e, const char *call, int line, const char *file)
    {
        if (e != cudaSuccess)
        {
            INFOE("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d",
                  call,
                  cudaGetErrorString(e),
                  cudaGetErrorName(e),
                  e, file, line);
            return false;
        }
        return true;
    }

    static std::string format(const char *fmt, ...)
    {
        va_list vl;
        va_start(vl, fmt);
        char buffer[2048];
        vsnprintf(buffer, sizeof(buffer), fmt, vl);
        return buffer;
    }
    bool check_device_id(int device_id)
    {
        int device_count = -1;
        checkRuntime(cudaGetDeviceCount(&device_count));
        if (device_id < 0 || device_id >= device_count)
        {
            INFOE("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }
    int current_device_id()
    {
        int device_id = 0;
        checkRuntime(cudaGetDevice(&device_id));
        return device_id;
    }
    std::string device_description()
    {
        cudaDeviceProp prop;
        size_t free_mem, total_mem;
        int device_id = 0;
        checkRuntime(cudaGetDevice(&device_id));
        checkRuntime(cudaGetDeviceProperties(&prop, device_id));
        checkRuntime(cudaMemGetInfo(&free_mem, &total_mem));

        return format(
            "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
            device_id, prop.name, prop.major, prop.minor,
            free_mem / 1024.0f / 1024.0f / 1024.0f,
            total_mem / 1024.0f / 1024.0f / 1024.0f);
    }

    AutoDevice::AutoDevice(int device_id)
    {
        cudaGetDevice(&old_);
        checkRuntime(cudaSetDevice(device_id));
    }
    AutoDevice::~AutoDevice()
    {
        checkRuntime(cudaSetDevice(old_));
    }
}