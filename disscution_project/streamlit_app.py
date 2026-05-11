"""Streamlit Dashboard — Plant Disease Detector Model Comparison.

Upload a leaf image and compare predictions from:
1. Scratch CNN (built from scratch — 4 Conv layers)
2. Transfer Learning (EfficientNet-B3)

Three tabs: individual model views + side-by-side comparison mode.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from disscution_project.CNN_from_scratch import build_scratch_cnn
from plant_disease_detector.model import build_model, load_checkpoint_weights
from plant_disease_detector.paths import load_label_map


# ══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Plant Disease Detector — Model Comparison",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — Professional Green/Nature Theme
# ══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
    }
    /* Header banner */
    .header-banner {
        background: linear-gradient(135deg, #1a5276 0%, #2ecc71 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .header-banner h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .header-banner p {
        margin: 0.25rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    /* Card containers */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .card h3 {
        margin-top: 0;
        font-size: 1.1rem;
        color: #1f2937;
    }
    /* Metric badge styling */
    .badge-green {
        background: #d1fae5;
        color: #065f46;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-orange {
        background: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-red {
        background: #fee2e2;
        color: #991b1b;
        padding: 0.2rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f3f4f6;
        padding: 0.25rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
    }
    /* Footer */
    .footer {
        text-align: center;
        color: #9ca3af;
        font-size: 0.8rem;
        padding: 1.5rem 0 0.5rem 0;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    /* Section divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #2ecc71, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════

IMAGE_SIZE = 380
NUM_CLASSES = 38
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model checkpoint paths
SCRATCH_CKPT = PROJECT_ROOT / "disscution_project" / "checkpoints" / "scratch_cnn_best.pt"
TRANSFER_CKPT = PROJECT_ROOT / "plant_disease_detector" / "checkpoints" / "best.pt"
TREATMENT_DB_PATH = PROJECT_ROOT / "data" / "treatment_db.json"

# Evaluation transform (no augmentation)
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


# ══════════════════════════════════════════════════════════════════════════
# GRAD-CAM UTILITIES
# ══════════════════════════════════════════════════════════════════════════

def compute_gradcam(
    model: nn.Module,
    tensor: torch.Tensor,
    target_layer: nn.Module,
    class_idx: int | None = None,
) -> np.ndarray:
    """Generic Grad-CAM for any model with a target layer."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def save_activation(m, i, o):
        activations.append(o.detach())
    def save_gradient(m, i, o):
        gradients.append(o[0].detach())

    fwd_hook = target_layer.register_forward_hook(save_activation)
    bwd_hook = target_layer.register_full_backward_hook(save_gradient)

    logits = model(tensor)
    if class_idx is None:
        class_idx = int(torch.argmax(logits, dim=1).item())
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()

    grad = gradients[0][0]
    act = activations[0][0]
    weights = torch.mean(grad, dim=(1, 2))
    cam = torch.sum(weights[:, None, None] * act, dim=0)
    cam = torch.relu(cam)
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.cpu().numpy()
    cam_resized = cv2.resize(cam_np, (IMAGE_SIZE, IMAGE_SIZE))

    fwd_hook.remove()
    bwd_hook.remove()
    return cam_resized


def overlay_heatmap(image_np: np.ndarray, cam: np.ndarray) -> np.ndarray:
    """Overlay heatmap on image. Returns RGB numpy array."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * image_np + 0.45 * heatmap).astype(np.uint8)
    return overlay


# ══════════════════════════════════════════════════════════════════════════
# SEVERITY LOGIC
# ══════════════════════════════════════════════════════════════════════════

def classify_severity(class_name: str, confidence: float) -> tuple[str, float, str]:
    """Classify severity as healthy/mild/moderate/severe. Returns (label, score, CSS class)."""
    if "healthy" in class_name.lower():
        return "Healthy", 0.0, "badge-green"
    score = round(max(0.0, min(1.0, 1.0 - confidence)), 3)
    if score < 0.33:
        return "Mild", score, "badge-green"
    elif score < 0.66:
        return "Moderate", score, "badge-orange"
    return "Severe", score, "badge-red"


def confidence_badge(confidence: float) -> str:
    """Return CSS class for confidence level."""
    if confidence > 0.85:
        return "badge-green"
    elif confidence > 0.6:
        return "badge-orange"
    return "badge-red"


def is_plant_healthy(class_name: str) -> bool:
    """True if label indicates a healthy plant (PlantVillage naming)."""
    return "healthy" in class_name.lower()


@st.cache_data
def load_treatment_db() -> dict[str, dict[str, str]]:
    """Load class key → treatment fields from data/treatment_db.json."""
    if not TREATMENT_DB_PATH.exists():
        return {}
    raw = json.loads(TREATMENT_DB_PATH.read_text(encoding="utf-8"))
    return {str(k): v for k, v in raw.items() if isinstance(v, dict)}


def render_treatment_suggestions(class_name: str, treatment_db: dict[str, dict[str, str]]) -> None:
    """Show curated treatment steps for diseased plants; light guidance if healthy."""
    st.markdown("**Treatment & management**")
    if not treatment_db:
        st.warning(
            f"Treatment database not found at `{TREATMENT_DB_PATH}`. "
            "Add `data/treatment_db.json` to enable suggestions."
        )
        return

    entry = treatment_db.get(class_name)
    if is_plant_healthy(class_name):
        st.success("No disease-specific treatment needed — prediction is a **healthy** class.")
        if entry:
            prev = (entry.get("prevention") or "").strip()
            if prev and prev.upper() != "N/A":
                st.markdown(prev)
        return

    if not entry:
        st.info(
            "No matching entry in the treatment database for this label. "
            "Confirm the diagnosis with a plant pathologist or extension service."
        )
        return

    st.caption("Reference only — follow product labels, local regulations, and expert advice.")

    sections: list[tuple[str, str]] = [
        ("immediate", "Immediate actions"),
        ("chemical", "Chemical options"),
        ("organic", "Organic / softer options"),
        ("prevention", "Prevention"),
    ]
    for key, title in sections:
        text = (entry.get(key) or "").strip()
        if not text or text.upper() == "N/A":
            continue
        st.markdown(f"**{title}**")
        st.markdown(text)


# ══════════════════════════════════════════════════════════════════════════
# MODEL LOADING (CACHED)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    """Load both models and label map. Cached so it only runs once."""
    label_map = load_label_map()

    # Scratch CNN
    scratch_model = build_scratch_cnn(num_classes=NUM_CLASSES).to(DEVICE)
    scratch_params = sum(p.numel() for p in scratch_model.parameters())
    if SCRATCH_CKPT.exists():
        ckpt = torch.load(SCRATCH_CKPT, map_location=DEVICE, weights_only=False)
        scratch_model.load_state_dict(ckpt["model_state"])
    scratch_model.eval()

    # Transfer Learning
    transfer_model = build_model(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
    transfer_params = sum(p.numel() for p in transfer_model.parameters())
    if TRANSFER_CKPT.exists():
        load_checkpoint_weights(transfer_model, str(TRANSFER_CKPT))
    transfer_model.eval()

    return scratch_model, transfer_model, label_map, scratch_params, transfer_params


# ══════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════

def predict(model: nn.Module, image: Image.Image) -> dict:
    """Run prediction on a PIL image. Returns top-3, timings, and class index."""
    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    start = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
    elapsed = time.perf_counter() - start

    top_probs, top_indices = torch.topk(probs, k=3, dim=1)
    top_probs = top_probs[0].cpu().tolist()
    top_indices = top_indices[0].cpu().tolist()

    return {
        "top_3": list(zip(top_indices, top_probs)),
        "top1_idx": top_indices[0],
        "top1_prob": top_probs[0],
        "inference_time_ms": round(elapsed * 1000, 1),
        "logits": logits,
        "tensor": tensor,
    }


# ══════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ══════════════════════════════════════════════════════════════════════════

def render_model_results(
    model_name: str,
    model_emoji: str,
    result: dict,
    params: int,
    label_map: dict,
    treatment_db: dict[str, dict[str, str]],
    gradcam_overlay: np.ndarray | None,
    original_image: Image.Image,
    col,
):
    """Render a single model's results inside a Streamlit column."""
    top1_label = label_map.get(result["top1_idx"], "Unknown")
    top1_conf = result["top1_prob"]
    severity, sev_score, sev_badge = classify_severity(top1_label, top1_conf)
    conf_badge = confidence_badge(top1_conf)

    # Determine heading color
    if top1_conf > 0.85:
        heading_color = "#065f46"
    elif top1_conf > 0.6:
        heading_color = "#92400e"
    else:
        heading_color = "#991b1b"

    with col:
        st.markdown(f"<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{model_emoji} {model_name}</h3>", unsafe_allow_html=True)
        st.caption(f"Parameters: {params:,}")

        # Prediction header
        st.markdown(
            f"<h1 style='margin:0; color: {heading_color};'>{top1_label}</h1>",
            unsafe_allow_html=True,
        )

        # Badges row
        st.markdown(
            f"<span class='{conf_badge}'>{(top1_conf*100):.1f}% confidence</span> "
            f"<span class='{sev_badge}' style='margin-left:0.5rem;'>{severity}</span>",
            unsafe_allow_html=True,
        )

        # Metrics row
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Confidence", f"{top1_conf*100:.1f}%")
        col_m2.metric("Severity", severity)
        col_m3.metric("Inference", f"{result['inference_time_ms']:.1f} ms")

        # Top-3 bar chart
        st.markdown("**Top-3 Predictions**")
        chart_data = {"Probability (%)": [p * 100 for _, p in result["top_3"]]}
        chart_labels = [
            label_map.get(idx, f"Class {idx}").replace("___", " - ").replace("_", " ")[:35]
            for idx, _ in result["top_3"]
        ]
        st.bar_chart(chart_data, height=180, use_container_width=True)
        max_prob = max(chart_data["Probability (%)"])
        for lbl, prob in zip(chart_labels, chart_data["Probability (%)"]):
            bar_width = max(prob, 2)
            bar_color = "#3b82f6" if prob == max_prob else "#9ca3af"
            st.markdown(
                f"<div style='display:flex; gap:0.5rem; font-size:0.8rem; margin-bottom:0.15rem;'>"
                f"<span style='width:65%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;'>{lbl}</span>"
                f"<div style='flex:1; background:#e5e7eb; border-radius:4px; height:14px; overflow:hidden;'>"
                f"<div style='width:{bar_width}%; background:{bar_color}; height:100%; border-radius:4px;'></div></div>"
                f"<span style='width:3rem; text-align:right;'>{prob:.0f}%</span></div>",
                unsafe_allow_html=True,
            )

        # Grad-CAM
        if gradcam_overlay is not None:
            st.markdown("**Grad-CAM Heatmap**")
            st.image(gradcam_overlay, caption=f"{model_name} — Activation Map", use_container_width=True)

        st.markdown("<hr style='border:none;border-top:1px solid #e5e7eb;margin:1rem 0;'/>", unsafe_allow_html=True)
        render_treatment_suggestions(top1_label, treatment_db)

        st.markdown("</div>", unsafe_allow_html=True)


def render_architecture_diagrams():
    """Show architecture diagrams when no image is uploaded."""
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Scratch CNN Architecture")
        st.code("""
Input(3×380×380)
├─ Conv2D(3→32) + ReLU + BN + MaxPool  [190×190×32]
├─ Conv2D(32→64) + ReLU + BN + MaxPool  [95×95×64]
├─ Conv2D(64→128) + ReLU + BN + MaxPool [47×47×128]
├─ Conv2D(128→256) + ReLU + BN + MaxPool[23×23×256]
├─ GlobalAvgPool → Flatten
├─ Dense(256) + ReLU + Dropout(0.5)
├─ Dense(128) + ReLU + Dropout(0.3)
└─ Dense(38) + Softmax
        """.strip())
        st.caption("**Total parameters:** 492,966 | **Built from scratch** — no pre-trained weights")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🚀 Transfer Learning — EfficientNet-B3")
        st.code("""
EfficientNet-B3 (pretrained on ImageNet)
├─ Phase 1: Backbone frozen
│  └─ Train only custom head
├─ Phase 2: Backbone unfrozen
│  └─ Fine-tune entire network
└─ Custom Classification Head:
   ├─ Linear(1536→512) + BatchNorm + ReLU
   ├─ Dropout(0.3)
   └─ Linear(512→38) + Softmax
        """.strip())
        st.caption("**Total parameters:** ~12,000,000 | **Pretrained** on ImageNet, fine-tuned on PlantVillage")
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("## 🌱 Model Dashboard")
st.sidebar.markdown("---")

# Load models
with st.spinner("Loading models..."):
    scratch_model, transfer_model, label_map, scratch_params, transfer_params = load_models()
    treatment_db = load_treatment_db()

st.sidebar.subheader("📦 Model Registry")
st.sidebar.markdown(
    f"**🧠 Scratch CNN**  \n"
    f"Params: `{scratch_params:,}`  \n"
    f"Status: {'✅ Trained' if SCRATCH_CKPT.exists() else '❌ Not trained'}  \n"
    f"Checkpoint: `{'Found' if SCRATCH_CKPT.exists() else 'Missing'}`"
)
st.sidebar.markdown(
    f"**🚀 EfficientNet-B3**  \n"
    f"Params: `{transfer_params:,}`  \n"
    f"Status: {'✅ Trained' if TRANSFER_CKPT.exists() else '❌ Not trained'}  \n"
    f"Checkpoint: `{'Found' if TRANSFER_CKPT.exists() else 'Missing'}`"
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Info")
st.sidebar.text(f"Device: {DEVICE}")
st.sidebar.text(f"Image Size: {IMAGE_SIZE}×{IMAGE_SIZE}")
st.sidebar.text(f"Classes: {NUM_CLASSES}")

st.sidebar.markdown("---")
st.sidebar.caption("Computer Vision & Image Processing  \nFinal Project | 2026")


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class='header-banner'>
        <h1>🌱 Plant Disease Detector</h1>
        <p>Compare predictions from a Scratch CNN vs Transfer Learning (EfficientNet-B3) — upload a leaf image to get started</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════
# IMAGE UPLOAD
# ══════════════════════════════════════════════════════════════════════════

upload_col, _, info_col = st.columns([3, 0.2, 1.5])

with upload_col:
    uploaded = st.file_uploader(
        "Upload a leaf image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a photo of a plant leaf to detect diseases. Accepted: JPG, PNG, WebP",
        label_visibility="collapsed",
    )

with info_col:
    if uploaded is not None:
        st.success(f"📎 **{uploaded.name}**")
        # Show file info
        file_size = len(uploaded.getvalue()) / 1024
        st.caption(f"Size: {file_size:.0f} KB | Type: {uploaded.type}")

if uploaded is None:
    # Show architecture diagrams when idle
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 📐 Model Architectures")
    st.info("👆 **Upload an image above** to see live predictions from both models")
    render_architecture_diagrams()
    st.stop()


# ══════════════════════════════════════════════════════════════════════════
# PROCESS IMAGE
# ══════════════════════════════════════════════════════════════════════════

image = Image.open(uploaded).convert("RGB")
image_np = np.array(image.resize((IMAGE_SIZE, IMAGE_SIZE)))

with st.spinner("🔬 Running predictions..."):
    scratch_result = predict(scratch_model, image)
    transfer_result = predict(transfer_model, image)

    # Grad-CAM
    scratch_cam = compute_gradcam(
        scratch_model, scratch_result["tensor"],
        target_layer=scratch_model.features[-2],
        class_idx=scratch_result["top1_idx"],
    )
    transfer_cam = compute_gradcam(
        transfer_model, transfer_result["tensor"],
        target_layer=transfer_model.backbone.conv_head,
        class_idx=transfer_result["top1_idx"],
    )
    scratch_overlay = overlay_heatmap(image_np, scratch_cam)
    transfer_overlay = overlay_heatmap(image_np, transfer_cam)

# ─── Original Image Preview ──────────────────────────────────────────

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
preview_col, _ = st.columns([1, 3])
with preview_col:
    st.image(image, caption="Uploaded Image", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TABS: Individual Model Views + Comparison
# ══════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs(["🧠 Scratch CNN", "🚀 Transfer Learning", "📊 Side-by-Side Comparison"])


# ─── TAB 1: Scratch CNN Only ─────────────────────────────────────────

with tab1:
    render_model_results(
        model_name="Scratch CNN",
        model_emoji="🧠",
        result=scratch_result,
        params=scratch_params,
        label_map=label_map,
        treatment_db=treatment_db,
        gradcam_overlay=scratch_overlay,
        original_image=image,
        col=st.container(),
    )


# ─── TAB 2: Transfer Learning Only ───────────────────────────────────

with tab2:
    render_model_results(
        model_name="EfficientNet-B3",
        model_emoji="🚀",
        result=transfer_result,
        params=transfer_params,
        label_map=label_map,
        treatment_db=treatment_db,
        gradcam_overlay=transfer_overlay,
        original_image=image,
        col=st.container(),
    )


# ─── TAB 3: Side-by-Side Comparison ──────────────────────────────────

with tab3:
    # Agreement banner
    models_agree = scratch_result["top1_idx"] == transfer_result["top1_idx"]
    if models_agree:
        st.success("✅ **Models Agree** — Both predict the same disease")
    else:
        st.warning("⚠️ **Models Disagree** — Predictions differ between models")

    # Side-by-side columns
    col_left, col_right = st.columns(2)

    # Extract labels for reuse
    scratch_label = label_map.get(scratch_result["top1_idx"], "Unknown")
    scratch_conf = scratch_result["top1_prob"]
    transfer_label = label_map.get(transfer_result["top1_idx"], "Unknown")
    transfer_conf = transfer_result["top1_prob"]

    # Heading colors
    if scratch_conf > 0.85:
        s_color = "#065f46"
    elif scratch_conf > 0.6:
        s_color = "#92400e"
    else:
        s_color = "#991b1b"
    if transfer_conf > 0.85:
        t_color = "#065f46"
    elif transfer_conf > 0.6:
        t_color = "#92400e"
    else:
        t_color = "#991b1b"

    # Left: Scratch CNN
    with col_left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🧠 Scratch CNN")
        st.caption(f"Parameters: {scratch_params:,}")
        sev_s, _, sev_b_s = classify_severity(scratch_label, scratch_conf)
        conf_b_s = confidence_badge(scratch_conf)
        st.markdown(
            f"<h2 style='color:{s_color}; margin:0;'>{scratch_label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='{conf_b_s}'>{(scratch_conf*100):.1f}%</span> "
            f"<span class='{sev_b_s}' style='margin-left:0.5rem;'>{sev_s}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Inference Time", f"{scratch_result['inference_time_ms']:.1f} ms")

        # Top-3
        st.markdown("**Top-3**")
        s_data = {"%": [p * 100 for _, p in scratch_result["top_3"]]}
        st.bar_chart(s_data, height=150, use_container_width=True)

        # Grad-CAM
        st.image(scratch_overlay, caption="Grad-CAM", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Right: Transfer Learning
    with col_right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🚀 EfficientNet-B3")
        st.caption(f"Parameters: {transfer_params:,}")
        sev_t, _, sev_b_t = classify_severity(transfer_label, transfer_conf)
        conf_b_t = confidence_badge(transfer_conf)
        st.markdown(
            f"<h2 style='color:{t_color}; margin:0;'>{transfer_label}</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<span class='{conf_b_t}'>{(transfer_conf*100):.1f}%</span> "
            f"<span class='{sev_b_t}' style='margin-left:0.5rem;'>{sev_t}</span>",
            unsafe_allow_html=True,
        )
        st.metric("Inference Time", f"{transfer_result['inference_time_ms']:.1f} ms")

        # Top-3
        st.markdown("**Top-3**")
        t_data = {"%": [p * 100 for _, p in transfer_result["top_3"]]}
        st.bar_chart(t_data, height=150, use_container_width=True)

        # Grad-CAM
        st.image(transfer_overlay, caption="Grad-CAM", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─── Treatment suggestions (linked to data/treatment_db.json) ─────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 💊 Treatment guidance")
    st.caption("Suggestions keyed to PlantVillage class names in `data/treatment_db.json`.")
    if models_agree:
        render_treatment_suggestions(scratch_label, treatment_db)
    else:
        tx_left, tx_right = st.columns(2)
        with tx_left:
            st.markdown(f"**Scratch CNN:** `{scratch_label}`")
            render_treatment_suggestions(scratch_label, treatment_db)
        with tx_right:
            st.markdown(f"**EfficientNet-B3:** `{transfer_label}`")
            render_treatment_suggestions(transfer_label, treatment_db)

    # ─── Comparison Metrics ─────────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("### 📊 Performance Comparison")

    col_c1, col_c2, col_c3, col_c4 = st.columns(4)

    with col_c1:
        st.metric("Parameters", f"{scratch_params:,}", f"{transfer_params - scratch_params:,} diff")
        st.caption("Scratch CNN / Transfer")

    with col_c2:
        st.metric("Inference Time", f"{scratch_result['inference_time_ms']:.1f} ms",
                  f"{transfer_result['inference_time_ms'] - scratch_result['inference_time_ms']:+.1f} ms")
        st.caption("Scratch CNN vs Transfer")

    with col_c3:
        conf_diff = transfer_conf - scratch_conf
        st.metric("Confidence Gap", f"{abs(conf_diff)*100:.1f}%",
                  f"{'Transfer higher' if conf_diff>0 else 'Scratch higher'}")
        st.caption("Difference in confidence")

    with col_c4:
        ratio = transfer_params / scratch_params
        st.metric("Size Ratio", f"{ratio:.1f}x", f"{'Transfer larger' if ratio>1 else 'Scratch larger'}")
        st.caption("Parameter count ratio")


# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <div class='footer'>
        🌱 Plant Disease Detector — Computer Vision & Image Processing Final Project<br>
        Built with Streamlit • PyTorch • EfficientNet-B3 • Custom CNN from Scratch<br>
        Dataset: PlantVillage (38 classes, Kaggle)
    </div>
    """,
    unsafe_allow_html=True,
)