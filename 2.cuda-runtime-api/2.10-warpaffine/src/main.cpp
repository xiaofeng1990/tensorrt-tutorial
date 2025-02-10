
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaimgproc.hpp>
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

void warp_affine_bilinear( // 声明
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value);

void warp_affine_bilinear_batch(
    uint8_t *src, int src_line_size, int src_frame_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_frame_size, int dst_width, int dst_height,
    uint8_t batch_size, uint8_t fill_value);
#if 0
cv::Mat warpaffine_to_center_align(const cv::Mat &image, const cv::Size &size)
{
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */

    cv::Mat output(size, CV_8UC3);
    uint8_t *psrc_device = nullptr;
    uint8_t *pdst_device = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMalloc(&psrc_device, src_size)); // 在GPU上开辟两块空间
    checkRuntime(cudaMalloc(&pdst_device, dst_size));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 搬运数据到GPU上

    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());

    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114);
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("gpu warp affine time %ld ms\n", timestamp2 - timestamp1);
    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    return output;
}

#else
cv::cuda::HostMem warpaffine_to_center_align(cv::Mat &image, const cv::Size &size)
{
    /*
       建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
            思路讲解：https://v.douyin.com/NhrNnVm/
            代码讲解: https://v.douyin.com/NhMv4nr/
    */

    // cv::Mat output(size, CV_8UC3);
    cv::cuda::HostMem output(size, CV_8UC3);

    uint8_t *psrc_device = nullptr;
    uint8_t *pdst_device = nullptr;
    uint8_t *pdst_host = nullptr;
    uint8_t *psrc_host = nullptr;
    size_t src_size = image.cols * image.rows * 3;
    size_t dst_size = size.width * size.height * 3;

    checkRuntime(cudaMallocHost(&pdst_host, dst_size));
    checkRuntime(cudaMallocHost(&psrc_host, src_size));

    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size));

    // memcpy(psrc_host, image.data, src_size);

    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    // 搬运数据到GPU上
    // checkRuntime(cudaMemcpy(psrc_device, psrc_host, src_size, cudaMemcpyHostToDevice));
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice));

    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114);

    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    // checkRuntime(cudaMemcpy(pdst_host, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("gpu warp affine time %ld ms\n", timestamp2 - timestamp1);
    // 检查核函数执行是否存在错误
    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));

    // cv::Mat output(size.height, size.width, CV_8UC3, (unsigned *)pdst_host);

    return output;
}
#endif

uint8_t **cuda_malloc_2d_host(uint8_t ***arr, const int m, const int n)
{
    cudaMallocHost(arr, m);
    if (!arr)
    {
        return nullptr;
    }
    for (int i = 0; i < m; i++)
    {
        cudaMallocHost(*arr + i, n);
        if (!(*arr + i))
        {
            return nullptr;
        }
    }
    return *arr;
}

uint8_t **cuda_malloc_2d_device(uint8_t ***arr, const int m, const int n)
{
    cudaMalloc(arr, m);
    if (!arr)
    {
        return nullptr;
    }
    for (int i = 0; i < m; i++)
    {
        cudaMalloc(*arr + i, n);
        if (!(*arr + i))
        {
            return nullptr;
        }
    }
    return *arr;
}

int warpaffine_to_center_align_batch(std::vector<std::string> &image_files, const cv::Size &size)
{
    size_t count = image_files.size();

    size_t src_width = 1280;
    size_t src_height = 720;
    uint8_t *psrc_device = nullptr;
    uint8_t *pdst_device = nullptr;
    uint8_t *pdst_host = nullptr;
    uint8_t *psrc_host = nullptr;
    size_t src_size = src_width * src_height * 3 * count;
    size_t src_fream_size = src_width * src_height * 3;
    size_t dst_size = size.width * size.height * 3 * count;
    size_t dst_fream_size = size.width * size.height * 3;

    checkRuntime(cudaMallocHost(&psrc_host, src_size));
    checkRuntime(cudaMallocHost(&pdst_host, dst_size));

    checkRuntime(cudaMalloc(&psrc_device, src_size));
    checkRuntime(cudaMalloc(&pdst_device, dst_size));

    uint8_t *psrc_host_index = psrc_host;

    for (size_t i = 0; i < count; i++)
    {
        // std::cout << image_files[i] << std::endl;
        cv::Mat image = cv::imread(image_files[i]);
        // cv::cuda::registerPageLocked(image);
        psrc_host_index = psrc_host + src_fream_size * i;
        memcpy(psrc_host_index, image.data, src_fream_size);
    }
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    checkRuntime(cudaMemcpy(psrc_device, psrc_host, src_size, cudaMemcpyHostToDevice));

    warp_affine_bilinear_batch(
        psrc_device, src_width * 3, src_width * src_height * 3, src_width, src_height,
        pdst_device, size.width * 3, size.width * size.height * 3, size.width, size.height, count, 114);

    checkRuntime(cudaDeviceSynchronize());
    checkRuntime(cudaMemcpy(pdst_host, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将预处理完的数据搬运回来
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("gpu batch warp affine %d image time %ld ms\n", count, timestamp2 - timestamp1);

    uint8_t *dst_index = pdst_host;

    for (size_t i = 0; i < count; i++)
    {
        dst_index = pdst_host + dst_fream_size * i;
        cv::Mat output = cv::Mat(size.height, size.width, CV_8UC3, (unsigned *)dst_index);
        std::string file_name = "output_" + std::to_string(i) + ".jpg";
        // std::cout << file_name << std::endl;
        cv::imwrite(file_name, output);
    }

    checkRuntime(cudaPeekAtLastError());

    checkRuntime(cudaFree(psrc_device));
    checkRuntime(cudaFree(pdst_device));
    checkRuntime(cudaFreeHost(pdst_host));
    checkRuntime(cudaFreeHost(psrc_host));
    return 0;
}

cv::Mat warp_affine_cpu(const cv::Mat &src)
{
    // warpAffine
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;

    float scale_x = input_width / (float)src.cols;
    float scale_y = input_height / (float)src.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6];
    float d2i[6];
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

cv::cuda::HostMem warp_affine_opencv_gpu(cv::Mat &src)
{
    // warpAffine
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;

    float scale_x = input_width / (float)src.cols;
    float scale_y = input_height / (float)src.rows;
    float scale = std::min(scale_x, scale_y);
    float i2d[6];
    float d2i[6];
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

    // 对图像做平移缩放旋转变换,可逆

    cv::cuda::setBufferPoolUsage(true);                                        // Tell OpenCV that we are going to utilize BufferPool
    cv::cuda::setBufferPoolConfig(cv::cuda::getDevice(), 1024 * 1024 * 64, 2); // Allocate 64 MB, 2 stacks (default is 10 MB, 5 stacks)

    cv::cuda::Stream stream1, stream2; // Each stream uses 1 stack
    cv::cuda::BufferPool pool1(stream1);

    cv::cuda::GpuMat g_src = pool1.getBuffer(src.size(), CV_8UC3); // 16MB
    auto systemtime = std::chrono::system_clock::now();
    uint64_t timestamp1(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());

    g_src.upload(src);
    cv::cuda::GpuMat g_warpt_image = pool1.getBuffer(input_height, input_width, CV_8UC3);

    cv::cuda::warpAffine(g_src, g_warpt_image, m2x3_i2d, g_warpt_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114), stream1);
    systemtime = std::chrono::system_clock::now();
    uint64_t timestamp2(std::chrono::duration_cast<std::chrono::milliseconds>(systemtime.time_since_epoch()).count());
    printf("cpu warp affine time %ld ms\n", timestamp2 - timestamp1);
    cv::cuda::HostMem warpt_image(input_height, input_width, CV_8UC3);
    g_warpt_image.download(warpt_image);

    return warpt_image;
}

int main()
{
    /*
    若有疑问，可点击抖音短视频辅助讲解(建议1.5倍速观看)
        https://v.douyin.com/NhMrb2A/
     */
    // int device_count = 1;
    // checkRuntime(cudaGetDeviceCount(&device_count));

    cv::Mat image = cv::imread("cat.jpg");

    // cv::cuda::registerPageLocked(image); // 按大小分配锁页内存

    auto output_cpu = warp_affine_cpu(image);

    cv::imwrite("output_cpu.jpg", output_cpu);

    // auto output = warpaffine_to_center_align(image, cv::Size(640, 640));

    // cv::imwrite("output.jpg", output);

    std::vector<std::string> image_files;
    size_t count = 12;
    for (size_t i = 1; i <= count; i++)
    {
        std::string image = "./test_images/" + std::to_string(i) + ".jpg";
        image_files.push_back(image);
    }
    warpaffine_to_center_align_batch(image_files, cv::Size(640, 640));

    printf("Done. save to output.jpg\n");

    // cv::cuda::unregisterPageLocked(image);
    return 0;
}
