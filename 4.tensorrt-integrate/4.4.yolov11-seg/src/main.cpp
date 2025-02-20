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

std::string join_dims(const std::vector<int> dims)
{
    std::stringstream output;
    char buf[64];
    const char *fmts[] = {"%d", " x %d"};
    for (int i = 0; i < dims.size(); ++i)
    {
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

std::string dims_str(const nvinfer1::Dims dims) { return join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims)); }

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

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
// hsv转bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    default:
        r = 1;
        g = 1;
        b = 1;
        break;
    }
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
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
            // else
            // {
            //     printf("%s: %s\n", severity_string(severity), msg);
            // }
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
// std::string root_path = "../4.2.yolov5-detect/data/";
std::string root_path = "./";

bool build_model()
{
    std::string engine_file = root_path + "yolo11n-seg_dynamic.engine";
    if (exists(engine_file))
    {
        printf("yolov5s.engine has exists.\n");
        return true;
    }
    TRTLogger logger;

    auto builder = make_shared(nvinfer1::createInferBuilder(logger));
    auto config = make_shared(builder->createBuilderConfig());
    auto network = make_shared(builder->createNetworkV2(1));

    config->setMaxWorkspaceSize(1 << 48);

    if (builder->platformHasFastFp16())
    {

        printf("Platform  support fast fp16\n");
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    else
    {
        printf("Platform support fp32\n");
        config->setFlag(nvinfer1::BuilderFlag::kTF32);
    }

    auto parser = make_shared(nvonnxparser::createParser(*network, logger));
    std::string onnx_file = root_path + "yolo11n-seg_dynamic.onnx";
    if (!parser->parseFromFile(onnx_file.c_str(), 1))
    {
        printf("Failed to parse yolo11n-seg_dynamic.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 5;
    int minBatchSize = 1;

    auto profile = builder->createOptimizationProfile();

    int input_number = network->getNbInputs();
    int output_number = network->getNbOutputs();
    std::cout << "input numbers: " << input_number << std::endl;
    std::cout << "output  numbers: " << output_number << std::endl;

    for (int i = 0; i < input_number; i++)
    {
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        std::cout << "input tensor name  " << input->getName() << std::endl;
        std::cout << "input tensor dims " << dims_str(input_dims).c_str() << std::endl;
        int batch = input_dims.d[0];
        if (batch <= 0)
        {
            input_dims.d[0] = minBatchSize;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
            input_dims.d[0] = maxBatchSize;
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
        }
    }
    config->addOptimizationProfile(profile);
    for (int i = 0; i < output_number; i++)
    {
        auto output = network->getOutput(i);
        auto output_dims = output->getDimensions();
        std::cout << "output tensor name  " << output->getName() << std::endl;
        std::cout << "output tensor dims  " << dims_str(output_dims).c_str() << std::endl;
    }

    auto engine = make_shared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_shared(engine->serialize());
    FILE *f = fopen(engine_file.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
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

    std::string engine_file = root_path + "yolo11n-seg_dynamic.engine";
    auto engine_data = load_file(engine_file);
    auto runtime = make_shared(nvinfer1::createInferRuntime(logger));
    auto engine = make_shared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    std::cout << "binding number  " << engine->getNbBindings() << std::endl;
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_shared(engine->createExecutionContext());

    int min_batch_size;

    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        auto dims = engine->getBindingDimensions(i);

        if (engine->bindingIsInput(i))
        {
            // 动态batch
            if (dims.d[0] <= 0)
            {
                int32_t profiles_number = engine->getNbOptimizationProfiles();
                auto dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batch_size = dims.d[0];
                dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
                opt_batch_size_ = dims.d[0];
                dims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batch_size_ = dims.d[0];
            }
            else
            {
                min_batch_size_ = dims.d[0];
                opt_batch_size_ = dims.d[0];
                max_batch_size_ = dims.d[0];
            }
            input_dims_ = dims;
            input_tensort_name_ = engine_->getBindingName(i);
            XT_LOGT(INFO, TAG, "input binding name  %s", input_tensort_name_.c_str());
            XT_LOGT(INFO, TAG, "input binding index  %d", engine_->getBindingIndex(engine_->getBindingName(i)));
            XT_LOGT(INFO, TAG, "input binding dim  %s", dims_str(dims).c_str());
        }
        else
        {
            output_dims_list_.push_back(dims);
            output_tensort_name_list_.push_back(engine_->getBindingName(i));

            XT_LOGT(INFO, TAG, "output binding name  %s", engine_->getBindingName(i));
            XT_LOGT(INFO, TAG, "output binding index  %d", engine_->getBindingIndex(engine_->getBindingName(i)));
            XT_LOGT(INFO, TAG, "output binding dim  %s", dims_str(dims).c_str());
        }
    }

    printf("engine->getName %s\n", engine->getName());
    auto dims = engine->getBindingDimensions(0);
    // int input_batch = dims.d[0];
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

    std::string image_file = root_path + "bus.jpg";
    auto image = cv::imread(image_file);
    // 通过双线性插值对图像进行resize
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6], d2i[6];
    /*
    M = [
            scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
            0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
            0,        0,                     1
        ]
    */
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * image.cols + input_width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d); // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i); // dst to image, 2x3 matrix

    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat input_image(input_height, input_width, CV_8UC3);
    // 对图像做平移缩放旋转变换,可逆
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    std::string warp_image_file = root_path + "input-image.jpg";
    cv::imwrite(warp_image_file, input_image);

    int image_area = input_image.cols * input_image.rows;
    unsigned char *pimage = input_image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }
    ///////////////////////////////////////////////////
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3输入，对应3x3输出
    auto output_dims = engine->getBindingDimensions(1);
    // int output_batch_size = output_dims.d[0];
    int output_batch_size = input_batch;
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];

    printf("output_batch_size %d, output_numbox %d, output_numprob %d \n", output_batch_size, output_numbox, output_numprob);
    int num_classes = output_numprob - 5;
    int output_numel = output_batch_size * output_numbox * output_numprob;
    float *output_data_host = nullptr;
    float *output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);
    float *bindings[] = {input_data_device, output_data_device};
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // decode box：从不同尺度下的预测狂还原到原输入图上(包括:预测框，类被概率，置信度）
    // yolov11 has an output of shape (batchSize,8400, 84) ( box[x,y,w,h] + Num classes)
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.45;
    for (int i = 0; i < output_numbox; ++i)
    {
        float *ptr = output_data_host + i * output_numprob;
        float *pclass = ptr + 4;
        int class_id = std::max_element(pclass, pclass + num_classes) - pclass;

        float confidence = pclass[class_id];
        if (confidence < confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx = ptr[0];
        float cy = ptr[1];
        float width = ptr[2];
        float height = ptr[3];

        // 预测框
        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 对应图上的位置
        float image_base_left = d2i[0] * left + d2i[2];
        float image_base_right = d2i[0] * right + d2i[2];
        float image_base_top = d2i[0] * top + d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)class_id, confidence});
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms非极大抑制
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float> &a, std::vector<float> &b)
              { return a[5] > b[5]; });
    std::vector<bool> remove_flags(bboxes.size());
    std::vector<std::vector<float>> box_result;
    box_result.reserve(bboxes.size());

    auto iou = [](const std::vector<float> &a, const std::vector<float> &b)
    {
        float cross_left = std::max(a[0], b[0]);
        float cross_top = std::max(a[1], b[1]);
        float cross_right = std::min(a[2], b[2]);
        float cross_bottom = std::min(a[3], b[3]);

        float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
        float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
        if (cross_area == 0 || union_area == 0)
            return 0.0f;
        return cross_area / union_area;
    };

    for (int i = 0; i < bboxes.size(); ++i)
    {
        if (remove_flags[i])
            continue;

        auto &ibox = bboxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < bboxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;

            auto &jbox = bboxes[j];
            if (ibox[4] == jbox[4])
            {
                // class matched
                if (iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for (int i = 0; i < box_result.size(); ++i)
    {
        auto &ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name = cocolabels[class_label];
        auto caption = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left - 3, top - 33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    std::string output_image_file = root_path + "output-image.jpg";
    cv::imwrite(output_image_file, image);

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
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