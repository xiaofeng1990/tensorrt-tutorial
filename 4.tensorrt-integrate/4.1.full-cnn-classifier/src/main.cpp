// tensorRT include
#include <NvInfer.h>
// onnx解析器的头文件
#include <NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
        return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
        return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "verbose";
    default:
        return "unknow";
    }
}
class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kINFO)
        {
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

bool exists(const std::string &path)
{
#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

template <typename _T>
std::shared_ptr<_T> make_shared(_T *ptr)
{
    return std::shared_ptr<_T>(ptr, [](_T *p)
                               { p->destroy(); });
}
bool build_model()
{
    if (exists("trtmodel.engine"))
    {
        printf("trtmodel.engine has exists.\n");
        return true;
    }
    TRTLogger logger;

    auto builder = make_shared(nvinfer1::createInferBuilder(logger));
    auto config = make_shared(builder->createBuilderConfig());
    auto network = make_shared(builder->createNetworkV2(1));
    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_shared(nvonnxparser::createParser(*network, logger));

    std::string root_path = "/home/wxf/workspace/tensorrt-tutorial/4.tensorrt-integrate/4.1.full-cnn-classifier/data/";

    std::string onnx_file = root_path + "classifier.onnx";

    if (!parser->parseFromFile(onnx_file.c_str(), 1))
    {
        printf("Failed to parse classifier.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    std::cout << "input dims";
    for (int i = 0; i < input_dims.nbDims; i++)
    {
        std::cout << input_dims.d[i] << " ";
    }
    std::cout << std::endl;
}
int main()
{
    build_model();
    return 0;
}