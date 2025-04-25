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
    // cx, cy, w, h, class*80, weight_32
    float *pitem = predict + (4 + 32 + num_classes) * position;

    // cx, cy, width, height, objness, classification*80
    float *class_confidence = pitem + 4;
    float confidence = 0;
    int label = 0;
    for (int i = 0; i < num_classes; i++, class_confidence++)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label = i;
        }
    }
    // 当前网格中有目标，且为某一个类别的的置信度
    if (confidence < confidence_threshold)
        return;
    // parray第一个元素记录保存了多少个box
    // 最大允许 max_objects 个目标，但是 parray[0] 最大可能大于max_objects 但实际只有max_objects个目标
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

    // left, top, right, bottom, confidence, class, keepflag, weight_row_index
    float *pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1;        // 1 = keep, 0 = ignore
    *pout_item++ = position; // weight_row_index
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

    // left, top, right, bottom, confidence, class, keepflag, weight_row_index
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

void decode_box_kernel_invoker(float *predict, int num_bboxes, int num_classes, float confidence_threshold,
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

static __global__ void decode_mask_kernel(float *output_device, float *mask_predict, int mask_dim, int mask_h, int mask_w, float *box_predict, float *boxes, int rows, int clos, int box_number)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    // int dz = blockDim.z * blockIdx.z + threadIdx.z;
    // 先确定box index
    // left, top, right, bottom, confidence, class, keepflag, weight_row_index
    // if (dz >= box_number)
    //     return;
    float *box = boxes + 1;
    float *output = output_device;
    int keepflag = box[6];
    if (!keepflag)
    {
        return;
    }

    float left = box[0] / 4;
    float top = box[1] / 4;
    float right = box[2] / 4;
    float bottom = box[3] / 4;
    int weight_row_index = box[7];
    if (dx < left || dx > right || dy < top || dy > bottom)
    {
        output[dy * mask_w + dx] = 0;
        return;
    }
    // cx, cy, w, h, class*80, weight*32
    float *mask_weights = box_predict + clos * weight_row_index + 84;
    float cumprod = 0;
    for (int ic = 0; ic < mask_dim; ic++)
    {
        float cval = mask_predict[ic * mask_h * mask_w + dy * mask_h + dx];
        float wval = mask_weights[ic];
        cumprod += cval * wval;
    }
    float alpha = 1.0f / (1.0f + exp(-cumprod)); // sigmoid
    // 获取box的内存
    if (alpha > 0.5)
        output[dy * mask_w + dx] = 255;
    else
        output[dy * mask_w + dx] = 0;
}

void decode_mask_kernel_invoker(float *output_device, float *mask_predict, int mask_dim, int mask_h, int mask_w, float *box_predict, float *boxes, int rows, int clos, int box_number)
{
    dim3 block_size(32, 32); // blocksize 最大就是1024，这里用2d来看更好理解
    dim3 grid_size((mask_h + 31) / 32, (mask_h + 31) / 32, box_number);

    decode_mask_kernel<<<grid_size, block_size, 0, nullptr>>>(output_device, mask_predict, mask_dim, mask_h, mask_w,
                                                              box_predict, boxes, rows, clos, box_number);
}