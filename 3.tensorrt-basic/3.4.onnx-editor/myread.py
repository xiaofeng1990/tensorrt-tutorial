import onnx
import numpy as np

model = onnx.load("yolov5s.onnx")

# for item in model.graph.initializer:
#     if item.name == "model.0.conv.conv.weight":
#         print("shape: ", item.dims)
#         weight = np.frombuffer(item.raw_data, dtype=np.float32)
#         print(weight)

for item in model.graph.node:
    if item.op_type == "Constant":
        if "346" in item.output:
            print(item.output)
