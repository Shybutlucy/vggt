import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
# dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
dtype = torch.float32  # 纯CPU模式，不使用GPU
# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
# image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]
image_names = ["F:\Job\Study\python\coding\vggt\examples\kitchen\images\00.png", "F:\Job\Study\python\coding\vggt\examples\kitchen\images\01.png", "F:\Job\Study\python\coding\vggt\examples\kitchen\images\02.png"]
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)