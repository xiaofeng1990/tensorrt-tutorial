#!/bin/bash

# 请修改protoc为你要使用的版本protoc
export LD_LIBRARY_PATH=/datav/software/anaconda3/lib/python3.9/site-packages/trtpy/trt8cuda112cudnn8/lib64
protoc=/datav/software/anaconda3/lib/python3.9/site-packages/trtpy/trt8cuda112cudnn8/bin/protoc

rm -rf pbout
mkdir -p pbout

$protoc onnx-ml.proto --cpp_out=pbout --python_out=pbout