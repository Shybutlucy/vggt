import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 只使用CPU不支持运行混合精度（autocast）

# 设备与精度
device = "cuda" if torch.cuda.is_available() else "cpu"
cap = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
dtype = torch.bfloat16 if cap >= 8 else torch.float16

# 加载模型（自动从Hugging Face下载权重）
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# 替换成你的图片路径
image_names = [
    "F:/Job/Study/python/coding/vggt/vggt/img/00.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/01.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/02.png"
]
images = load_and_preprocess_images(image_names).to(device)

# 推理
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        pred = model(images)

# import pdb;pdb.set_trace()
# 输出结果
print("相机参数:", pred["camera"])
print("深度图 shape:", pred["depth_map"].shape)
print("3D点云 shape:", pred["point_map"].shape)