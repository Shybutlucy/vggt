import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os
import numpy as np
from PIL import Image

# ======================== 强制 CPU ========================
device = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ======================== 加载模型 ========================
print("加载模型...")
model = VGGT()
model = model.to(device)
model.eval()

# ======================== 图片路径 ========================
image_names = [
    "F:/Job/Study/python/coding/vggt/vggt/img/00.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/01.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/02.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/03.png",
    "F:/Job/Study/python/coding/vggt/vggt/img/04.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/05.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/06.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/07.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/08.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/09.png",
    # "F:/Job/Study/python/coding/vggt/vggt/img/10.png"
]

# 检查图片
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

# 真实输出字段
depth = pred["depth"]
world_points = pred["world_points"]

print("\n📊 数据形状：")
print("深度图 shape:", depth.shape)
print("3D点云 shape:", world_points.shape)

# ======================== 输出路径 ========================
output_dir = r"F:\Job\Study\python\coding\vggt\vggt\img\output"
os.makedirs(output_dir, exist_ok=True)

# ======================== ✅ 修复：正确保存深度图 ========================
# 把 5 维 → 2 维图片 (H, W)
depth_np = depth[0, 0, :, :, 0].cpu().numpy()  # 关键修复！
depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255
depth_np = depth_np.astype(np.uint8)

depth_save_path = os.path.join(output_dir, "depth.png")
Image.fromarray(depth_np).save(depth_save_path)
print(f"\n✅ 深度图已保存：{depth_save_path}")

# ======================== ✅ 修复：正确保存3D点云 ========================
points_np = world_points[0, 0].reshape(-1, 3).cpu().numpy()  # 关键修复！
ply_save_path = os.path.join(output_dir, "3d_points.ply")

with open(ply_save_path, "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n" % len(points_np))
    for p in points_np:
        f.write(f"{p[0]} {p[1]} {p[2]}\n")

print(f"✅ 3D点云已保存：{ply_save_path}")