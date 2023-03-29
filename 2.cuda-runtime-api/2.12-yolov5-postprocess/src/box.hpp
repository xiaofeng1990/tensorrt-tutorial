#ifndef __BOX_HPP_
#define __BOX_HPP_
struct Box
{
    float left;
    float top;
    float right;
    float bottom;
    float confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label) : left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}
};

#endif //__BOX_HPP_H_