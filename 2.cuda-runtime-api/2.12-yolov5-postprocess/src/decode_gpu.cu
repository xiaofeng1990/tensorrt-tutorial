#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include <device_launch_parameters.h>

static __global__ void decode_kernel(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                                     float *invert_affine_matrix, float *parray, int max_objects, int NUM_BOX_ELEMENT)

{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    // 获得当前线程的box信息
    float *pitem = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    // cx, cy, width, height, objness, classification*80
    float *class_confidence = pitem + 5;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    // 当前网格中有目标，且为某一个类别的的置信度
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;
    // parray第一个元素记录保存了多少个box
    int index = atomicAdd(parray, 1);
    if (index >= max_objects)
        return;
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // left, top, right, bottom, confidence, class, keepflag
    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __global__ void decode_kernel_batch(float *predict, size_t predict_size, int num_bboxes, int num_classes,
                                           float confidence_threshold, float *invert_affine_matrix, float *parray, size_t output_size,
                                           size_t batch, int max_objects, int NUM_BOX_ELEMENT)

{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= num_bboxes || dy >= batch)
        return;

    // 获得当前线程的box信息
    float *predict_index = predict + predict_size * dy;
    float *parray_index = parray + output_size * dy;

    float *pitem = predict_index + (5 + num_classes) * dx;
    float objectness = pitem[4];
    if (objectness < confidence_threshold)
        return;

    // cx, cy, width, height, objness, classification*80
    float *class_confidence = pitem + 5;
    float confidence = *class_confidence++;
    int label = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    // 当前网格中有目标，且为某一个类别的的置信度
    confidence *= objectness;
    if (confidence < confidence_threshold)
        return;
    // parray第一个元素记录保存了多少个box
    int index = atomicAdd(parray_index, 1);
    if (index >= max_objects)
        return;
    float cx = *pitem++;
    float cy = *pitem++;
    float width = *pitem++;
    float height = *pitem++;
    float left = cx - width * 0.5f;
    float top = cy - height * 0.5f;
    float right = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;

    // left, top, right, bottom, confidence, class, keepflag
    float *pout_item = parray_index + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom,
    float bleft, float btop, float bright, float bbottom)
{

    float cleft = max(aleft, bleft);
    float ctop = max(atop, btop);
    float cright = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float *bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT)
{

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for (int i = 0; i < count; ++i)
    {
        // 不和自己比较，如果是不同的类别也不比较
        float *pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static __global__ void fast_nms_kernel_batch(float *bboxes, int max_objects, float threshold, size_t output_size, int batch, int NUM_BOX_ELEMENT)
{

    int dx = (blockDim.x * blockIdx.x + threadIdx.x);
    int dy = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (dx >= count || dy >= batch)
        return;

    float *bboxes_index = bboxes + output_size * dy;
    // left, top, right, bottom, confidence, class, keepflag
    float *pcurrent = bboxes_index + 1 + dx * NUM_BOX_ELEMENT;

    for (int i = 0; i < count; ++i)
    {
        // 不和自己比较，如果是不同的类别也不比较
        float *pitem = bboxes_index + 1 + i * NUM_BOX_ELEMENT;
        if (i == dx || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < dx)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void decode_kernel_invoker(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
                           float nms_threshold, float *invert_affine_matrix, float *parray, int max_objects,
                           int NUM_BOX_ELEMENT, cudaStream_t stream)
{
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    decode_kernel<<<grid, block, 0, stream>>>(predict, num_bboxes, num_classes, confidence_threshold,
                                              invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT);

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}

void decode_kernel_invoker_batch(float *predict, size_t predict_size, int num_bboxes, int num_classes, float confidence_threshold,
                                 float nms_threshold, float *invert_affine_matrix, float *parray, size_t output_size, int max_objects,
                                 int NUM_BOX_ELEMENT, size_t batch, cudaStream_t stream)
{
    auto block_size = num_bboxes > 1024 ? 1024 : num_bboxes;

    dim3 grid((num_bboxes + block_size - 1) / block_size, batch);
    dim3 block(block_size);

    // printf("grid size = %d * %d * %d \n", grid.x, grid.y, grid.z);
    // printf("block size = %d * %d * %d \n", block.x, block.y, block.z);
    // decode_kernel_batch<<<grid, block, 0, stream>>>(predict, predict_size, num_bboxes, num_classes, confidence_threshold,
    //                                                 invert_affine_matrix, parray, output_size, batch, max_objects, NUM_BOX_ELEMENT);
    decode_kernel_batch<<<grid, block, 0, nullptr>>>(predict, predict_size, num_bboxes, num_classes, confidence_threshold,
                                                     invert_affine_matrix, parray, output_size, batch, max_objects, NUM_BOX_ELEMENT);

    block_size = max_objects > 1024 ? 1024 : max_objects;
    dim3 grid_nms((max_objects + block_size - 1) / block_size, batch);
    dim3 block_nms(block_size);
    // printf("grid_nms size = %d * %d * %d \n", grid_nms.x, grid_nms.y, grid_nms.z);
    // printf("block_nms size = %d * %d * %d \n", block_nms.x, block_nms.y, block_nms.z);
    fast_nms_kernel_batch<<<grid_nms, block_nms, 0, nullptr>>>(parray, max_objects, nms_threshold, output_size, batch, NUM_BOX_ELEMENT);
}