import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os

# ======================== 强制 CPU ========================
device = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ======================== 关键：本地模型 ========================
# 不从 HF 下载！直接用本地自带权重，彻底解决 401 / 418 / 网络错误
print("加载本地模型...")
model = VGGT()  # 不加 from_pretrained，用本地定义
model = model.to(device)
model.eval()

# ======================== 图片路径 ========================
image_names = [
    "F:/Job/Study/python/coding/vggt/vggt/img/00.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/01.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/02.png",
]

# 检查路径是否存在
for p in image_names:
    print("✅ 存在" if os.path.exists(p) else "❌ 不存在", p)

# 加载图片
images = load_and_preprocess_images(image_names).to(device)

# ======================== 推理 ========================
print("开始推理...")
with torch.no_grad():
    pred = model(images)

# ======================== 输出结果 ========================
print("\n" + "="*50)
print("✅ 运行成功！")
print("输出 keys:", list(pred.keys()))