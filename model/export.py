# https://github.com/Deci-AI/super-gradients
from super_gradients.training import models

net = models.get("yolo_nas_s", pretrained_weights="coco")
models.convert_to_onnx(model=net, input_shape=(3,640,640), out_path="yolo_nas_s.onnx",
    torch_onnx_export_kwargs={"opset_version": 12})