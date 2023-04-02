// cuda runtime 头文件
#include <cuda_runtime.h>
#include <stdio.h>
#include <string.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

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

int main()
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
