import io
import base64
import torch
import torch.nn as nn
import timm
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only, restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CBAM Modules ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.shared_mlp(self.avg_pool(x))
        max_ = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg + max_)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel = ChannelAttention(in_planes, ratio)
        self.spatial = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel(x)
        x = x * self.spatial(x)
        return x

class EfficientNetCBAM(nn.Module):
    def __init__(self, num_classes=5):
        super(EfficientNetCBAM, self).__init__()
        self.backbone = timm.create_model("efficientnet_b4", pretrained=True, features_only=False)
        self.cbam = CBAM(in_planes=1792)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1792, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        features = self.cbam(features)
        self._features_for_cam = features  # used for Grad-CAM
        pooled = self.pool(features).view(x.size(0), -1)
        return self.fc(pooled)

# --- Load Model ---
device = torch.device("cpu")
model = EfficientNetCBAM(num_classes=5)
model.load_state_dict(torch.load(r"D:\Liver-Fiberosis\model\efficientnet_cbam_fibrosis_aug.pth", map_location=device))
model.to(device).eval()

# --- Transform ---
transform = A.Compose([
    A.Resize(380, 380),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

# --- Prediction Route ---
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    original_image = image.copy()
    image_np = np.array(image)
    augmented = transform(image=image_np)
    image_tensor = augmented["image"].unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_class = output.argmax(dim=1).item()

    # GradCAM
    target_layer = model.cbam
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(pred_class)])[0]

    # Overlay GradCAM
    rgb_img = np.array(original_image.resize((380, 380))).astype(np.float32) / 255.0
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # Convert to base64
    result_image = Image.fromarray(cam_image)
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return JSONResponse(content={
        "predicted_class": int(pred_class),
        "gradcam_image": f"data:image/jpeg;base64,{img_str}"
    })
