"""
DeepDetect Sentinel — Model Inference Module
TripleStreamDetector with GradCAM forensic heatmap analysis.
Trained on Celeb-DF dataset. Video-only model.
"""

import os
import io
import zipfile
import base64
import numpy as np
import cv2
import torch
import torch.nn as nn

try:
    from facenet_pytorch import MTCNN
except ImportError:
    MTCNN = None

import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

# ─── Configuration ──────────────────────────────────────────
NUM_FRAMES = 8
IMG_SIZE   = 224
THRESHOLD  = 0.25
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_triple_stream_v10.pth")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Model Architecture ─────────────────────────────────────

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
        blurred = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        return x - blurred

    def forward(self, frames):
        """frames: [B, T, C, H, W] → [B, out_dim]"""
        B, T, C, H, W = frames.shape
        flat = frames.view(B * T, C, H, W)
        hp   = self.high_pass(flat)
        feat = self.proj(self.conv(hp))
        return feat.view(B, T, -1).mean(dim=1)


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

        frame_feat = self.backbone(video.view(B * T, C, H, W))
        frame_feat = frame_feat.view(B, T, -1)

        # ── Stream 1: Spatial ──
        attn_w = torch.softmax(self.spatial_attn(frame_feat).squeeze(-1), dim=1)
        spatial_feat = (frame_feat * attn_w.unsqueeze(-1)).sum(dim=1)

        # ── Stream 2: Temporal ──
        cls_tokens    = self.temporal_cls.expand(B, -1, -1)
        temporal_out  = self.temporal_encoder(torch.cat([cls_tokens, frame_feat], dim=1))
        temporal_feat = temporal_out[:, 0, :]

        # ── Stream 3: Frequency ──
        freq_feat = self.freq_stream(video)

        # ── Fusion ──
        combined = torch.cat([spatial_feat, temporal_feat, freq_feat], dim=1)
        return self.fusion(combined).squeeze(-1)


# ─── Preprocessing ──────────────────────────────────────────

_mtcnn = None

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
    Returns list of PIL Images (face crops), base64-encoded thumbnails, and video metadata.
    """
    global _mtcnn
    
    cap          = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = round(total_frames / fps, 2) if fps > 0 else 0.0

    if total_frames <= 0:
        cap.release()
        raise ValueError("Could not read video frames. Invalid or corrupt video file.")

    if MTCNN is not None and _mtcnn is None:
        _mtcnn = MTCNN(keep_all=False, device=DEVICE)

    indices    = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames     = []
    thumbnails = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img   = Image.fromarray(frame_rgb)
            
            # --- Extract Face ---
            if _mtcnn is not None:
                boxes, _ = _mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w_box, h_box = x2 - x1, y2 - y1
                    margin_w, margin_h = int(w_box * 0.3), int(h_box * 0.3)
                    
                    x1, y1 = max(0, x1 - margin_w), max(0, y1 - margin_h)
                    x2, y2 = min(frame_rgb.shape[1], x2 + margin_w), min(frame_rgb.shape[0], y2 + margin_h)
                    
                    face_img = frame_rgb[y1:y2, x1:x2]
                    if face_img.size > 0:
                        pil_img = Image.fromarray(face_img)

            frames.append(pil_img)

            # Generate thumbnail from the potentially cropped image
            face_img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # Use square-ish size if it's a cropped face, else 16:9
            thumb_size = (240, 240) if _mtcnn is not None and pil_img.size[0] < pil_img.size[1]*1.5 else (240, 135)
            thumb = cv2.resize(face_img_bgr, thumb_size)
            _, buffer = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
            b64 = base64.b64encode(buffer).decode('utf-8')
            thumbnails.append(f"data:image/jpeg;base64,{b64}")
        else:
            if frames:
                frames.append(frames[-1].copy())
                thumbnails.append(thumbnails[-1])
            else:
                black = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
                frames.append(black)
                thumbnails.append("")

    cap.release()
    meta = {
        "fps":          round(fps, 2),
        "duration":     duration,
        "resolution":   f"{width}\u00d7{height}",
        "total_frames": total_frames,
    }
    return frames, thumbnails, meta


def preprocess_frames(frames):
    """Transform PIL frames to model input tensor."""
    tensors = [_transform(f) for f in frames]
    batch   = torch.stack(tensors).unsqueeze(0)
    return batch


# ─── Model Loading ──────────────────────────────────────────
_model = None


def load_model():
    """Load the trained model from disk."""
    global _model
    if _model is not None:
        return _model

    print(f"[DeepDetect] Loading model from {MODEL_PATH} on {DEVICE}...")

    model = TripleStreamDetector()

    if os.path.exists(MODEL_PATH):
        if MODEL_PATH.endswith('.zip'):
            with zipfile.ZipFile(MODEL_PATH, 'r') as zf:
                pth_files = [f for f in zf.namelist() if f.endswith('.pth')]
                if pth_files:
                    with zf.open(pth_files[0]) as f:
                        buffer     = io.BytesIO(f.read())
                        state_dict = torch.load(buffer, map_location=DEVICE, weights_only=False)
                else:
                    raise FileNotFoundError("No .pth file found in zip archive")
        else:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

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


# ─── GradCAM ────────────────────────────────────────────────

def compute_frame_gradcam(model, pil_frame, device):
    """
    Compute multi-layer GradCAM heatmap for face-manipulation localization.

    Three key improvements over a naive single-layer approach:

    1. FULL 3-stream backward signal
       The original code only used the spatial head with temporal/frequency
       streams zeroed out.  Using the complete fusion logit means all learned
       features (spatial + temporal + frequency) contribute to the gradient,
       giving much stronger and more accurate spatial discriminability.

    2. Logit backward (no sigmoid saturation)
       sigmoid'(x) ≈ 0 when |x| >> 0, so backpropagating through sigmoid
       kills gradients whenever the model is confident.  Backpropagating
       through the raw logit (sigmoid skipped) avoids this entirely.

    3. Multi-layer CAM aggregation (last 4 blocks)
       The last transformer block is very high-level / globally pooled.
       Middle-late blocks still carry local spatial structure (face region
       edges, texture).  Summing normalized CAMs from the last 4 blocks
       produces dramatically better face localization for deepfake detection.

    Returns:
      heatmap_b64 (str)       : data-URL JPEG of the blended overlay.
      per_frame_prob (float)  : sigmoid fake probability for this frame.
    """
    tensor   = _transform(pil_frame).unsqueeze(0).to(device)   # [1, 3, 224, 224]
    orig_rgb = np.array(pil_frame.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)

    # ── Register forward hooks on the last 4 transformer blocks ─────────────
    # We aggregate their CAMs to capture both local and global spatial cues.
    num_aggregate = 4
    block_saves   = []
    hooks         = []

    for block in model.backbone.blocks[-num_aggregate:]:
        sv = {}

        def _make_hook(storage):
            def _h(module, inp, out):
                out.retain_grad()
                storage['act'] = out
            return _h

        hooks.append(block.register_forward_hook(_make_hook(sv)))
        block_saves.append(sv)

    frame_prob = 0.5
    blended    = orig_bgr.copy()

    try:
        with torch.enable_grad():
            model.zero_grad()

            # ── Full 3-stream forward on a single replicated frame ───────────
            feat     = model.backbone(tensor)        # [1, D=384]
            feat_seq = feat.unsqueeze(1)             # [1, 1, D]

            # Stream 1 – Spatial
            attn_w  = torch.softmax(
                model.spatial_attn(feat_seq).squeeze(-1), dim=1
            )
            spatial = (feat_seq * attn_w.unsqueeze(-1)).sum(dim=1)  # [1, D]

            # Stream 2 – Temporal  (single-frame: feed feat as the sequence)
            cls_tok  = model.temporal_cls.expand(1, -1, -1)         # [1, 1, D]
            temp_out = model.temporal_encoder(
                torch.cat([cls_tok, feat_seq], dim=1)
            )
            temporal = temp_out[:, 0, :]                            # [1, D]

            # Stream 3 – Frequency  (expects [B, T, C, H, W])
            freq_in = tensor.unsqueeze(1)                           # [1, 1, 3, 224, 224]
            freq    = model.freq_stream(freq_in)                    # [1, 128]

            # Full fusion → raw logit (NOT sigmoid – avoids saturation)
            combined   = torch.cat([spatial, temporal, freq], dim=1)  # [1, 896]
            logit      = model.fusion(combined)                        # [1, 1]
            frame_prob = torch.sigmoid(logit).item()

            # Gradients through the raw logit – magnitude is constant,
            # no vanishing-gradient near extremes.
            logit.sum().backward()

        # ── Aggregate multi-block Grad-CAM ───────────────────────────────────
        combined_cam = None

        for sv in block_saves:
            act_t = sv.get('act')
            if act_t is None or act_t.grad is None:
                continue

            # Patch tokens only (exclude the CLS at position 0)
            act  = act_t[0, 1:, :].detach()         # [N_patches, D]
            grad = act_t.grad[0, 1:, :].detach()    # [N_patches, D]

            # Standard Grad-CAM: per-channel importance = mean gradient over patches
            alpha_k = grad.mean(dim=0)               # [D]
            cam_i   = F.relu((act * alpha_k).sum(dim=-1))  # [N_patches]

            # Fallback: gradient magnitude if CAM is degenerate
            if cam_i.max() < 1e-9:
                cam_i = (act * grad).abs().sum(dim=-1)

            # Normalize each layer to [0,1] before summing
            cam_max = cam_i.max()
            if cam_max > 1e-9:
                cam_i = cam_i / cam_max

            combined_cam = cam_i if combined_cam is None else combined_cam + cam_i

        # ── Build the visual overlay ─────────────────────────────────────────
        if combined_cam is not None and combined_cam.max() > 1e-9:
            side   = int(combined_cam.numel() ** 0.5)          # 14 for patch16
            cam_np = combined_cam.cpu().numpy().reshape(side, side).astype(np.float32)

            # Power-transform normalization: spreads mid-range activations,
            # making face regions pop out more clearly.
            cam_np = np.power(np.clip(cam_np, 0, None), 0.5)
            cmin, cmax = cam_np.min(), cam_np.max()
            if cmax - cmin > 1e-8:
                cam_np = (cam_np - cmin) / (cmax - cmin)
            else:
                cam_np = np.zeros_like(cam_np)

            # Upsample to 224×224 with cubic interpolation
            cam_up = cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE),
                                interpolation=cv2.INTER_CUBIC)

            # Jet colormap: blue (clean) → green → red (manipulated)
            heatmap_color = cv2.applyColorMap(
                np.uint8(255 * cam_up), cv2.COLORMAP_JET
            )

            # Intensity-weighted alpha blend:
            #   low-activation pixels → show original frame
            #   high-activation pixels → show heatmap strongly
            # This is better than a fixed alpha because it preserves the
            # face structure in low-suspicion areas.
            alpha_map = cam_up[:, :, np.newaxis].astype(np.float32)   # [H,W,1]
            blended   = (
                orig_bgr.astype(np.float32) * (1.0 - 0.70 * alpha_map) +
                heatmap_color.astype(np.float32) * 0.70 * alpha_map
            ).clip(0, 255).astype(np.uint8)

    except Exception as exc:
        print(f"[GradCAM] Frame error: {exc}")
        import traceback; traceback.print_exc()
        blended = orig_bgr

    finally:
        for h in hooks:
            h.remove()

    _, buf = cv2.imencode('.jpg', blended, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return f"data:image/jpeg;base64,{base64.b64encode(buf).decode()}", frame_prob


# ─── Inference ──────────────────────────────────────────────


def analyze_video(video_path):
    """
    Run deepfake detection on a video file.
    Returns a rich dict with:
      - verdict, probability, confidence, risk level
      - per-stream scores (spatial, temporal, frequency)
      - GradCAM heatmap overlays for every frame
      - per-frame fake probabilities
      - most/least confident (most/least manipulated) frame info
      - video metadata
    """
    model = load_model()

    # ── Extract frames ───────────────────────────────────────
    frames, thumbnails, video_meta = extract_frames(video_path, NUM_FRAMES)
    input_tensor = preprocess_frames(frames).to(DEVICE)

    # ── Main Inference (no_grad for full-video score) ────────
    with torch.no_grad():
        B, T, C, H, W = input_tensor.shape

        frame_feat = model.backbone(input_tensor.view(B * T, C, H, W))
        frame_feat = frame_feat.view(B, T, -1)

        # Stream 1: Spatial
        attn_w       = torch.softmax(model.spatial_attn(frame_feat).squeeze(-1), dim=1)
        spatial_feat = (frame_feat * attn_w.unsqueeze(-1)).sum(dim=1)
        spatial_score = torch.sigmoid(model.fusion(torch.cat([
            spatial_feat,
            torch.zeros_like(spatial_feat),
            torch.zeros(B, 128, device=DEVICE)
        ], dim=1))).item()

        # Stream 2: Temporal
        cls_tokens    = model.temporal_cls.expand(B, -1, -1)
        temporal_out  = model.temporal_encoder(torch.cat([cls_tokens, frame_feat], dim=1))
        temporal_feat = temporal_out[:, 0, :]
        temporal_score = torch.sigmoid(model.fusion(torch.cat([
            torch.zeros_like(temporal_feat),
            temporal_feat,
            torch.zeros(B, 128, device=DEVICE)
        ], dim=1))).item()

        # Stream 3: Frequency
        freq_feat  = model.freq_stream(input_tensor)
        freq_score = torch.sigmoid(model.fusion(torch.cat([
            torch.zeros(B, 384, device=DEVICE),
            torch.zeros(B, 384, device=DEVICE),
            freq_feat
        ], dim=1))).item()

        # Full combined inference
        combined  = torch.cat([spatial_feat, temporal_feat, freq_feat], dim=1)
        logits    = model.fusion(combined)
        prob_fake = torch.sigmoid(logits).item()

    # ── Per-Frame GradCAM ────────────────────────────────────
    print(f"[DeepDetect] Computing GradCAM for {len(frames)} frames on {DEVICE}...")
    heatmap_frames = []
    frame_probs    = []

    for i, frame in enumerate(frames):
        hmap_b64, fprob = compute_frame_gradcam(model, frame, DEVICE)
        heatmap_frames.append(hmap_b64)
        frame_probs.append(round(fprob, 6))
        print(f"  [GradCAM] Frame {i+1}/{len(frames)}: p(fake)={fprob:.4f}")

    # ── Most / Least Manipulated Frames ─────────────────────
    most_idx  = int(np.argmax(frame_probs))
    least_idx = int(np.argmin(frame_probs))

    most_confident_frame = {
        "index":        most_idx,
        "frame_number": most_idx + 1,
        "prob":         frame_probs[most_idx],
        "heatmap":      heatmap_frames[most_idx],
        "thumbnail":    thumbnails[most_idx],
    }
    least_confident_frame = {
        "index":        least_idx,
        "frame_number": least_idx + 1,
        "prob":         frame_probs[least_idx],
        "heatmap":      heatmap_frames[least_idx],
        "thumbnail":    thumbnails[least_idx],
    }

    # ── Final Classification ─────────────────────────────────
    prediction = "fake" if prob_fake > THRESHOLD else "real"
    confidence = (round(prob_fake * 100, 2) if prediction == "fake"
                  else round((1 - prob_fake) * 100, 2))

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
        # ── Verdict ──
        "prediction":   prediction,
        "prob_fake":    round(prob_fake, 6),
        "confidence":   confidence,
        "threshold":    THRESHOLD,
        "risk_level":   risk_level,
        "num_frames":   len(frames),
        "device":       str(DEVICE),
        # ── Stream scores ──
        "stream_scores": {
            "spatial":   round(spatial_score * 100, 2),
            "temporal":  round(temporal_score * 100, 2),
            "frequency": round(freq_score * 100, 2),
        },
        # ── Frame data ──
        "thumbnails":            thumbnails,
        "heatmap_frames":        heatmap_frames,
        "frame_probs":           frame_probs,
        "most_confident_frame":  most_confident_frame,
        "least_confident_frame": least_confident_frame,
        # ── Video metadata ──
        "video_fps":          video_meta["fps"],
        "video_duration":     video_meta["duration"],
        "video_resolution":   video_meta["resolution"],
        "video_total_frames": video_meta["total_frames"],
    }
