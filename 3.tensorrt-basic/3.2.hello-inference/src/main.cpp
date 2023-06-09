// cuda runtime 头文件
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
// system include
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>

class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
        {
            printf("%d: %s\n", severity, msg);
        }
    }
};

nvinfer1::Weights make_weights(float *ptr, int n)
{
    nvinfer1::Weights w;
    w.count = n;
    w.type = nvinfer1::DataType::kFLOAT;
    w.values = ptr;

    return w;
}

bool build_model()
{
    // logger是必要的，用来捕捉warning和info等
    TRTLogger logger;
    // ----------------------------- 1. 定义 builder, config 和network -----------------------------
    // 使用一个builder来builde一个网络
    nvinfer1::IBuilder *builder = nvinfer1::createInferBuilder(logger);
    // 每一个builder需要一个config来定义网络参数，
    nvinfer1::IBuilderConfig *config = builder->createBuilderConfig();
    // builder需要一个network来定义网络结构
    nvinfer1::INetworkDefinition *network = builder->createNetworkV2(1);

    // 构建模型
    /*
        Network definition
        image
          |
        linear(fully connected) input=3, output=2, bias=True w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
          |
        sigmoid
          |
         prob
    */
    // ----------------------------- 2. 输入，模型结构和输出的基本信息 -----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb
    float layer1_bias_values[] = {0.3, 0.8};
    // 输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
    nvinfer1::ITensor *input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1));
    nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6);
    nvinfer1::Weights layer1_bias = make_weights(layer1_bias_values, 2);

    // 添加全连接层
    auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);
    // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用
    auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

    // 将prob标记为输出
    network->markOutput(*prob->getOutput(0));
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f); // 256Mib
    config->setMaxWorkspaceSize(1 << 28);
    // 推理时 batchSize = 1
    builder->setMaxBatchSize(1);

    // ----------------------------- 3. 生成engine模型文件 -----------------------------
    // TensorRT 7.1.0版本已弃用buildCudaEngine方法，统一使用buildEngineWithConfig方法
    nvinfer1::ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return -1;
    }
    // ----------------------------- 4. 序列化模型文件并存储 -----------------------------
    // 将模型序列化，并储存为文件
    nvinfer1::IHostMemory *model_data = engine->serialize();
    FILE *f = fopen("trtmodel.engine", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    model_data->destroy();
    engine->destroy();
    network->destroy();
    config->destroy();
    builder->destroy();
    printf("Done.\n");

    return 0;
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

void inference()
{
    // ------------------------------ 1. 准备模型并加载   ----------------------------
    TRTLogger logger;
    auto engine_data = load_file("trtmodel.engine");
    // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    nvinfer1::IExecutionContext *execution_context = engine->createExecutionContext();
    cudaStream_t stream = nullptr;
    // 创建CUDA流，以确定这个batch的推理是独立的
    cudaStreamCreate(&stream);
    /*
      Network definition:

      image
        |
      linear (fully connected)  input = 3, output = 2, bias = True     w=[[1.0, 2.0, 0.5], [0.1, 0.2, 0.5]], b=[0.3, 0.8]
        |
      sigmoid
        |
      prob
    */
    // ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
    float input_data_host[] = {1, 2, 3};
    float *input_data_device = nullptr;
    float output_data_host[2];
    float *output_data_device = nullptr;

    cudaMalloc(&input_data_device, sizeof(input_data_host));
    cudaMalloc(&output_data_device, sizeof(output_data_host));

    cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
    // 用一个指针数组指定input和output在gpu中的指针。
    float *bindings[] = {input_data_device, output_data_device};

    // ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
    // 异步推理
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    printf("output_data_host = %f, %f\n", output_data_host[0], output_data_host[1]);

    // ------------------------------ 4. 释放内存 ----------------------------
    printf("Clean memory\n");
    cudaStreamDestroy(stream);
    execution_context->destroy();
    engine->destroy();
    runtime->destroy();
    // ------------------------------ 5. 手动推理进行验证 ----------------------------
    const int num_input = 3;
    const int num_output = 2;
    float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5};
    float layer1_bias_values[] = {0.3, 0.8};

    printf("手动验证计算结果：\n");
    for (int io = 0; io < num_output; ++io)
    {
        float output_host = layer1_bias_values[io];
        for (int ii = 0; ii < num_input; ++ii)
        {
            output_host += layer1_weight_values[io * num_input + ii] * input_data_host[ii];
        }

        // sigmoid
        float prob = 1 / (1 + exp(-output_host));
        printf("output_prob[%d] = %f\n", io, prob);
    }
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
