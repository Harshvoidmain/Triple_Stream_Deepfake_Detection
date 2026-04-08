"""
DeepDetect Sentinel — Model Inference Module
EfficientNet B3 with Attention Pooling for deepfake video detection.
Trained on Celeb-DF dataset. Video-only model.
"""

import os
import io
import zipfile
import tempfile
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ─── Configuration ──────────────────────────────────────────
NUM_FRAMES = 8
IMG_SIZE = 224
THRESHOLD = 0.25
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_triple_stream_v10.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Architecture ────────────────────────────────────
class FrequencyStream(nn.Module):
    """Stream 3: High-pass filter → small CNN → frequency-domain features."""
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(128, out_dim)

    def high_pass(self, x):
        """x: [B, 3, H, W] → high-frequency residual."""
        blurred = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        return x - blurred

    def forward(self, frames):
        """frames: [B, T, C, H, W] → [B, out_dim]"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        hp   = self.high_pass(flat)
        feat = self.proj(self.conv(hp))   # [B*T, out_dim]
        return feat.view(B, T, -1).mean(dim=1)  # temporal mean → [B, out_dim]


class TripleStreamDetector(nn.Module):
    """Triple-Stream Deepfake Detector with enhanced regularization.
       Stream 1 (Spatial)  : DeiT-Small ViT + attention pooling
       Stream 2 (Temporal) : Same ViT features + Transformer encoder
       Stream 3 (Frequency): High-pass filter + small CNN
    """
    def __init__(self, backbone='deit_small_patch16_224', dropout=0.4,
                 freq_dim=128, drop_path_rate=0.2):
        super().__init__()
        # ── Shared ViT backbone (Stream 1 & 2) ──
        self.backbone = timm.create_model(
            backbone, pretrained=False, num_classes=0,
            drop_path_rate=drop_path_rate
        )
        feat_dim = self.backbone.num_features  # 384 for DeiT-Small

        # ── Stream 1: Spatial attention pooling ──
        self.spatial_attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.Tanh(),
            nn.Linear(feat_dim // 4, 1)
        )

        # ── Stream 2: Temporal transformer encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=6, dim_feedforward=feat_dim * 2,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_cls = nn.Parameter(torch.randn(1, 1, feat_dim) * 0.02)

        # ── Stream 3: Frequency domain ──
        self.freq_stream = FrequencyStream(out_dim=freq_dim)

        # ── Fusion head ──
        total_dim = feat_dim + feat_dim + freq_dim  # 384 + 384 + 128 = 896
        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Dropout(dropout),
            nn.Linear(total_dim, total_dim // 2),
            nn.GELU(),
            nn.LayerNorm(total_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(total_dim // 2, 1)
        )

    def forward(self, video):
        B, T, C, H, W = video.shape

        # Extract per-frame ViT features
        frame_feat = self.backbone(video.view(B * T, C, H, W))
        frame_feat = frame_feat.view(B, T, -1)

        # ── Stream 1: Spatial ──
        attn_w = torch.softmax(self.spatial_attn(frame_feat).squeeze(-1), dim=1)
        spatial_feat = (frame_feat * attn_w.unsqueeze(-1)).sum(dim=1)

        # ── Stream 2: Temporal ──
        cls_tokens = self.temporal_cls.expand(B, -1, -1)
        temporal_input = torch.cat([cls_tokens, frame_feat], dim=1)
        temporal_out   = self.temporal_encoder(temporal_input)
        temporal_feat  = temporal_out[:, 0, :]

        # ── Stream 3: Frequency ──
        freq_feat = self.freq_stream(video)

        # ── Fusion ──
        combined = torch.cat([spatial_feat, temporal_feat, freq_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


# ─── Preprocessing ──────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def extract_frames(video_path, num_frames=NUM_FRAMES):
    """Extract evenly spaced frames from a video file.
    Returns list of PIL Images, list of base64-encoded thumbnails, and video metadata.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = round(total_frames / fps, 2) if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video frames. Invalid or corrupt video file.")

    # Evenly spaced frame indices
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    thumbnails = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)

            # Create thumbnail for UI (larger for better visibility)
            thumb = cv2.resize(frame, (240, 135))
            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            thumbnails.append(f"data:image/jpeg;base64,{b64}")
        else:
            # If frame read fails, duplicate last frame or use black
            if frames:
                frames.append(frames[-1].copy())
                thumbnails.append(thumbnails[-1])
            else:
                black = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
                frames.append(black)
                thumbnails.append("")

    cap.release()
    meta = {
        "fps": round(fps, 2),
        "duration": duration,
        "resolution": f"{width}×{height}",
        "total_frames": total_frames,
    }
    return frames, thumbnails, meta


def preprocess_frames(frames):
    """Transform PIL frames to model input tensor."""
    tensors = [_transform(f) for f in frames]
    # Stack: (num_frames, C, H, W) -> add batch dim: (1, num_frames, C, H, W)
    batch = torch.stack(tensors).unsqueeze(0)
    return batch


# ─── Model Loading ──────────────────────────────────────────
_model = None


def load_model():
    """Load the trained model from zip file."""
    global _model
    if _model is not None:
        return _model

    print(f"[DeepDetect] Loading model from {MODEL_PATH} on {DEVICE}...")

    model = TripleStreamDetector()

    if os.path.exists(MODEL_PATH):
        # Handle zip-compressed weights
        if MODEL_PATH.endswith('.zip'):
            with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
                # Find the .pth file inside
                pth_files = [f for f in zf.namelist() if f.endswith('.pth')]
                if pth_files:
                    with zf.open(pth_files[0]) as f:
                        buffer = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=DEVICE, weights_only=False)
                else:
                    raise FileNotFoundError("No .pth file found in zip archive")
        else:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

        # Handle state dict wrapped in 'model_state_dict' or 'model' key
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

        model.load_state_dict(state_dict, strict=False)
        print("[DeepDetect] Model weights loaded successfully.")
    else:
        print(f"[DeepDetect] WARNING: Model file not found at {MODEL_PATH}. Using random weights.")

    model.to(DEVICE)
    model.eval()
    _model = model
    return model


# ─── Inference ──────────────────────────────────────────────
def analyze_video(video_path):
    """
    Run deepfake detection on a video file.
    Returns dict with prediction, probability, threshold, frame thumbnails, and rich metadata.
    """
    model = load_model()

    # Extract and preprocess frames
    frames, thumbnails, video_meta = extract_frames(video_path, NUM_FRAMES)
    input_tensor = preprocess_frames(frames).to(DEVICE)

    # Run inference with per-stream partial scores
    with torch.no_grad():
        B, T, C, H, W = input_tensor.shape

        # Get per-stream features for individual stream scores
        frame_feat = model.backbone(input_tensor.view(B * T, C, H, W))
        frame_feat = frame_feat.view(B, T, -1)

        # Stream 1: Spatial
        attn_w = torch.softmax(model.spatial_attn(frame_feat).squeeze(-1), dim=1)
        spatial_feat = (frame_feat * attn_w.unsqueeze(-1)).sum(dim=1)
        spatial_score = torch.sigmoid(model.fusion(torch.cat([
            spatial_feat,
            torch.zeros_like(spatial_feat),
            torch.zeros(B, 128, device=DEVICE)
        ], dim=1))).item()

        # Stream 2: Temporal
        cls_tokens = model.temporal_cls.expand(B, -1, -1)
        temporal_out = model.temporal_encoder(torch.cat([cls_tokens, frame_feat], dim=1))
        temporal_feat = temporal_out[:, 0, :]
        temporal_score = torch.sigmoid(model.fusion(torch.cat([
            torch.zeros_like(temporal_feat),
            temporal_feat,
            torch.zeros(B, 128, device=DEVICE)
        ], dim=1))).item()

        # Stream 3: Frequency
        freq_feat = model.freq_stream(input_tensor)
        freq_score = torch.sigmoid(model.fusion(torch.cat([
            torch.zeros(B, 384, device=DEVICE),
            torch.zeros(B, 384, device=DEVICE),
            freq_feat
        ], dim=1))).item()

        # Full combined inference
        combined = torch.cat([spatial_feat, temporal_feat, freq_feat], dim=1)
        logits = model.fusion(combined)
        prob_fake = torch.sigmoid(logits).item()

    # Binary decision using tuned threshold
    prediction = "fake" if prob_fake > THRESHOLD else "real"
    confidence = round(prob_fake * 100, 2) if prediction == "fake" else round((1 - prob_fake) * 100, 2)

    # Risk level classification
    if prob_fake >= 0.75:
        risk_level = "CRITICAL"
    elif prob_fake >= 0.50:
        risk_level = "HIGH"
    elif prob_fake >= 0.25:
        risk_level = "MEDIUM"
    elif prob_fake >= 0.10:
        risk_level = "LOW"
    else:
        risk_level = "MINIMAL"

    return {
        "prediction": prediction,
        "prob_fake": round(prob_fake, 6),
        "confidence": confidence,
        "threshold": THRESHOLD,
        "num_frames": len(frames),
        "thumbnails": thumbnails,
        "risk_level": risk_level,
        "device": str(DEVICE),
        "stream_scores": {
            "spatial": round(spatial_score * 100, 2),
            "temporal": round(temporal_score * 100, 2),
            "frequency": round(freq_score * 100, 2),
        },
        "video_fps": video_meta["fps"],
        "video_duration": video_meta["duration"],
        "video_resolution": video_meta["resolution"],
        "video_total_frames": video_meta["total_frames"],
    }
