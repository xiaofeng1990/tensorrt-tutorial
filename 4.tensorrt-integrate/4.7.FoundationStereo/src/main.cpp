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
#include "NvInferPlugin.h"
#include <opencv2/opencv.hpp>
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
void decode_box_kernel_invoker(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                               float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects,
                               int NUM_BOX_ELEMENT, cudaStream_t stream);
void decode_mask_kernel_invoker(float *output_device, float *mask_predict, int mask_dim, int mask_h, int mask_w,
                                float *box_predict, float *boxes, int rows, int clos, int box_number);

struct Box
{
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;
    std::vector<float> weight;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label) : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}
};
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
std::string root_path = "/home/xintent/workspace/wxf/tensorrt-tutorial/4.tensorrt-integrate/build/";

// std::string root_path = "/home/wxf/workspace/tensorrt-tutorial/4.tensorrt-integrate/build/";

bool build_model()
{
    std::string engine_file = root_path + "foundation_stereo_23-51-11.engine";
    if (exists(engine_file))
    {
        printf("foundation_stereo_23-51-11.engine has exists.\n");
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
    std::string onnx_file = root_path + "foundation_stereo_23-51-11.onnx";
    if (!parser->parseFromFile(onnx_file.c_str(), 1))
    {
        printf("Failed to parsefoundation_stereo_23-51-11.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }

    int maxBatchSize = 1;
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
            config->addOptimizationProfile(profile);
        }
    }

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

cv::Mat warp_affine_cpu(const cv::Mat &src, float *d2i)
{
    // warpAffine
    int input_channel = 3;
    int input_height = 480;
    int input_width = 640;

    float scale_x = input_width / (float)src.cols;
    float scale_y = input_height / (float)src.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6];
    /*
        M = [
           scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
           0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
           0,        0,                     1
        ]
    */
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * src.cols + input_width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * src.rows + input_height + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d); // image to dst(network), 2x3 matrix
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i); // dst to image, 2x3 matrix
    // 获得逆矩阵
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);

    cv::Mat warpt_image(input_height, input_width, CV_8UC3);
    // 对图像做平移缩放旋转变换,可逆
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    cv::warpAffine(src, warpt_image, m2x3_i2d, warpt_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
                   cv::Scalar::all(114));
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("cpu warp affine 1 image time %ld ms\n", timestamp2 - timestamp1);

    return warpt_image;
}
float iou(const Box &a, const Box &b)
{
    float cross_left = std::max(a.left, b.left);
    float cross_top = std::max(a.top, b.top);
    float cross_right = std::min(a.right, b.right);
    float cross_bottom = std::min(a.bottom, b.bottom);

    float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
    float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) +
                       std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
    if (cross_top == 0 || union_area == 0)
        return 0.0f;

    return cross_area / union_area;
}
std::vector<Box> decode_box_cpu(float *predict, int rows, int clos, float *d2i, float confidence_threshold = 0.25f, float nms_threshold = 0.45f)
{
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::microseconds>(systemtime.time_since_epoch()).count());

    // cx, cy, width, height, objness, classification*80
    // 一行是85列
    std::vector<Box> boxes;
    int num_classes = clos - 36;

    // 第一个循环，根据置信度挑选box
    for (int i = 0; i < rows; i++)
    {
        // 获得每一行的首地址
        float *pitem = predict + i * clos;

        // 获取类别置信度的首地址
        float *pclass = pitem + 4;
        // std::max_element 返回从pclass到pclass+num_classes中最大值的地址，
        // 减去 pclass 后就是索引
        int class_id = std::max_element(pclass, pclass + num_classes) - pclass;
        float confidence = pclass[class_id];
        if (confidence < confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx = pitem[0];
        float cy = pitem[1];
        float width = pitem[2];
        float height = pitem[3];

        // 预测框
        Box box;
        box.left = cx - width * 0.5;
        box.top = cy - height * 0.5;
        box.right = cx + width * 0.5;
        box.bottom = cy + height * 0.5;
        box.confidence = confidence;
        box.label = class_id;
        // memcpy(box.weight, pitem + 84, 32 * sizeof(float));
        std::vector<float> mask(pitem + 84, pitem + 84 + 32);
        box.weight = mask;
        // 对应图上的位置
        // float image_base_left = d2i[0] * left + d2i[2];
        // float image_base_right = d2i[0] * right + d2i[2];
        // float image_base_top = d2i[0] * top + d2i[5];
        // float image_base_bottom = d2i[0] * bottom + d2i[5];
        // boxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)class_id, confidence});
        boxes.push_back(box);
    }
    // 对所有box根据置信度排序
    std::sort(boxes.begin(), boxes.end(), [](Box &a, Box &b)
              { return a.confidence > b.confidence; });
    // 记录box是否被删除，被删除为true
    std::vector<bool> remove_flags(boxes.size());
    // 保存box
    std::vector<Box> box_result;
    box_result.reserve(boxes.size());

    for (int i = 0; i < boxes.size(); i++)
    {
        if (remove_flags[i])
            continue;
        auto &ibox = boxes[i];
        box_result.emplace_back(ibox);
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (remove_flags[j])
                continue;
            auto &jbox = boxes[j];
            if (ibox.label == jbox.label)
            {
                if (iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::microseconds>(systemtime.time_since_epoch()).count());

    printf("cpu yolov5 postprocess %ld ns\n", timestamp2 - timestamp1);

    return box_result;
}

void inference()
{
    TRTLogger logger;
    nvinfer1::ILogger *gLogger;
    initLibNvInferPlugins(gLogger, "");
    std::string engine_file = root_path + "foundation_stereo.engine";
    auto engine_data = load_file(engine_file);
    auto runtime = make_shared(nvinfer1::createInferRuntime(logger));
    auto engine = make_shared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    printf("engine_data.size() %d.\n", engine_data.size());
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
    int opt_batch_size;
    int max_batch_size;
    std::vector<std::string> input_tensort_name_list;
    std::vector<std::string> output_tensort_name_list;
    int input_channel;
    int input_height;
    int input_width;
    for (int i = 0; i < engine->getNbBindings(); i++)
    {
        auto dims = engine->getBindingDimensions(i);

        if (engine->bindingIsInput(i))
        {
            std::cout << "input binding name " << engine->getBindingName(i) << std::endl;
            input_tensort_name_list.push_back(engine->getBindingName(i));

            std::cout << "input binding index  " << engine->getBindingIndex(engine->getBindingName(i)) << std::endl;
            std::cout << "input binding dim  " << dims_str(dims) << std::endl;
            // 动态batch
            if (dims.d[0] <= 0)
            {
                int32_t profiles_number = engine->getNbOptimizationProfiles();
                auto dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);
                min_batch_size = dims.d[0];
                std::cout << "min tensor dims " << dims_str(dims) << std::endl;
                std::cout << "min_batch_size  " << min_batch_size << std::endl;
                dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kOPT);
                opt_batch_size = dims.d[0];
                std::cout << "opt tensor dims " << dims_str(dims) << std::endl;
                std::cout << "opt_batch_size  " << opt_batch_size << std::endl;
                dims = engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
                max_batch_size = dims.d[0];
                std::cout << "max tensor dims " << dims_str(dims) << std::endl;
                std::cout << "max_batch_size  " << max_batch_size << std::endl;
            }
            else
            {
                min_batch_size = dims.d[0];
                opt_batch_size = dims.d[0];
                max_batch_size = dims.d[0];
            }
            input_channel = dims.d[1];
            input_height = dims.d[2];
            input_width = dims.d[3];
        }
        else
        {

            std::cout << "output binding name " << engine->getBindingName(i) << std::endl;
            std::cout << "output binding index  " << engine->getBindingIndex(engine->getBindingName(i)) << std::endl;
            std::cout << "output binding dim  " << dims_str(dims) << std::endl;
            output_tensort_name_list.push_back(engine->getBindingName(i));
        }
    }

    // 分配input 内存
    int input_binding_index_0 = engine->getBindingIndex(input_tensort_name_list[0].c_str());
    nvinfer1::DataType type = engine->getBindingDataType(input_binding_index_0);

    std::cout << "input tensort type  " << (int)type << std::endl;

    int input_numel = min_batch_size * input_channel * input_height * input_width;
    float *input_data_host_left = nullptr;
    float *input_data_device_left = nullptr;
    float *input_data_host_right = nullptr;
    float *input_data_device_right = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host_left, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device_left, input_numel * sizeof(float)));
    checkRuntime(cudaMallocHost(&input_data_host_right, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device_right, input_numel * sizeof(float)));

    // output0 index 2 box
    int output_binding_index = engine->getBindingIndex(output_tensort_name_list[0].c_str());
    type = engine->getBindingDataType(output_binding_index);

    std::cout << "output tensort type  " << (int)type << std::endl;

    auto output_dims = engine->getBindingDimensions(output_binding_index);
    int output_batch_size = output_dims.d[0];
    int output_channle = output_dims.d[1];
    int output_height = output_dims.d[2];
    int output_width = output_dims.d[3];
    printf("output_batch_size %d, output_channle %d, output_height %d ,output_width %d\n", output_batch_size, output_channle, output_height, output_width);
    // int num_classes = output_numprob - 4;
    int output_numel = output_batch_size * output_channle * output_height * output_width;
    float *output_data_host = nullptr;
    float *output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    std::string image_file_left = root_path + "left.png";
    auto image = cv::imread(image_file_left);
    float d2i[6];
    auto warp_image = warp_affine_cpu(image, d2i);
    int image_area = warp_image.cols * warp_image.rows;
    unsigned char *pimage = warp_image.data;
    float *phost_b = input_data_host_left + image_area * 0;
    float *phost_g = input_data_host_left + image_area * 1;
    float *phost_r = input_data_host_left + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    std::cout << "copy left input data to device  " << std::endl;
    if (warp_image.data)
        checkRuntime(cudaMemcpyAsync(input_data_device_left, input_data_host_left, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    else
        std::cout << "input is null " << std::endl;

    std::string image_file_right = root_path + "right.png";
    auto image_right = cv::imread(image_file_right);
    float d2i_right[6];
    auto warp_image_right = warp_affine_cpu(image_right, d2i_right);
    int image_area_right = warp_image_right.cols * warp_image_right.rows;
    unsigned char *pimage_right = warp_image_right.data;
    float *phost_b_right = input_data_host_right + image_area_right * 0;
    float *phost_g_right = input_data_host_right + image_area_right * 1;
    float *phost_r_right = input_data_host_right + image_area_right * 2;
    for (int i = 0; i < image_area_right; ++i, pimage_right += 3)
    {
        // 注意这里的顺序rgb调换了
        *phost_r_right++ = pimage_right[0] / 255.0f;
        *phost_g_right++ = pimage_right[1] / 255.0f;
        *phost_b_right++ = pimage_right[2] / 255.0f;
    }

    std::cout << "copy right input data to device  " << std::endl;
    if (warp_image.data)
        checkRuntime(cudaMemcpyAsync(input_data_device_right, input_data_host_right, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    else
        std::cout << "input is null " << std::endl;

    float *bindings[] = {input_data_device_left, input_data_device_right, output_data_device};
    std::cout << "infer " << std::endl;
    auto start_time_ = std::chrono::high_resolution_clock::now();
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr);
    std::cout << "copy output to host " << std::endl;
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));
    auto stop_time_ = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_time_ - start_time_);

    std::cout << "duration " << duration.count() << std::endl;
}
// https://github.com/NVlabs/FoundationStereo/tree/master
int main()
{
    // if (!build_model())
    // {
    //     return -1;
    // }
    inference();
    return 0;
}