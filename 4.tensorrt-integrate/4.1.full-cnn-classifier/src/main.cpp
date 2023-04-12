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
std::string root_path = "/home/wxf/workspace/tensorrt-tutorial/4.tensorrt-integrate/4.1.full-cnn-classifier/data/";

bool build_model()
{
    std::string engine_file = root_path + "classifier.engine";
    if (exists(engine_file))
    {
        printf("classifier.engine has exists.\n");
        return true;
    }
    TRTLogger logger;

    auto builder = make_shared(nvinfer1::createInferBuilder(logger));
    auto config = make_shared(builder->createBuilderConfig());
    auto network = make_shared(builder->createNetworkV2(1));
    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_shared(nvonnxparser::createParser(*network, logger));

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
    std::cout << "input tensor name: " << input_tensor->getName() << " dims: ";
    for (int i = 0; i < input_dims.nbDims; i++)
    {
        std::cout << input_dims.d[i] << " ";
    }
    std::cout << std::endl;
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);
    auto engine = make_shared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }
    // 将模型序列化，并储存为文件 nvinfer1::IHostMemory
    auto model_data = make_shared(engine->serialize());

    FILE *f = fopen(engine_file.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);
    // 卸载顺序按照构建顺序倒序
    printf("Done.\n");
    return true;
}

std::vector<unsigned char> load_file(const std::string &file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();
    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char *)&data[0], length);
    }
    in.close();

    return data;
}
std::vector<std::string> load_labels(const char *file)
{
    std::vector<std::string> lines;

    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        printf("open %d failed.\n", file);
        return lines;
    }

    std::string line;
    while (getline(in, line))
    {
        lines.push_back(line);
    }
    in.close();
    return lines;
}

void inference()
{
    TRTLogger logger;
    std::string engine_file = root_path + "classifier.engine";
    auto engine_data = load_file(engine_file);

    auto runtime = make_shared(nvinfer1::createInferRuntime(logger));

    auto engine = make_shared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_shared(engine->createExecutionContext());

    printf("engine->getBindingName(0) %s\n", engine->getBindingName(0));
    printf("engine->getName %s\n", engine->getName());
    auto dims = engine->getBindingDimensions(0);
    int input_batch = 1;
    int input_channel = dims.d[1];
    int input_height = dims.d[2];
    int input_width = dims.d[3];
    printf("input_batch %d input_channel %d input_height %d input_width %d\n",
           input_batch, input_channel, input_height, input_width);
    int input_numel = input_batch * input_channel * input_height * input_width;
    float *input_data_host = nullptr;
    float *input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    std::string image_file = root_path + "dog.jpg";
    auto image = cv::imread(image_file);
    float mean[] = {0.406, 0.456, 0.485};
    float std[] = {0.225, 0.224, 0.229};

    cv::resize(image, image, cv::Size(input_width, input_height));
    int image_area = image.cols * image.rows;
    unsigned char *pimage = image.data;

    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 3;

    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = (pimage[0] / 255.0f - mean[0]) / std[0];
        *phost_g++ = (pimage[1] / 255.0f - mean[1]) / std[1];
        *phost_b++ = (pimage[2] / 255.0f - mean[2]) / std[2];
    }
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto dims_output = engine->getBindingDimensions(1);
    printf("dims_output.nbDims %d \n", dims_output.nbDims);
    printf("dims_output.d[0] %d \n", dims_output.d[0]);
    printf("dims_output.d[1] %d \n", dims_output.d[1]);

    const int num_classes = 1000;
    float output_data_host[num_classes];
    float *output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(output_data_host)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    input_dims.d[0] = input_batch;
    // 设置当前推理时，input大小
    execution_context->setBindingDimensions(0, input_dims);

    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    float *prob = output_data_host;
    int predict_label = std::max_element(prob, prob + num_classes) - prob; // 确定预测类别的下标

    std::string labels_file = root_path + "labels.imagenet.txt";
    auto labels = load_labels(labels_file.c_str());
    auto predict_name = labels[predict_label];
    float confidence = prob[predict_label]; // 获得预测值的置信度
    printf("Predict: %s, confidence = %f, label = %d\n", predict_name.c_str(), confidence, predict_label);
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main()
{
    if (!build_model())
    {
        return -1;
    }
    inference();
    return 0;
}