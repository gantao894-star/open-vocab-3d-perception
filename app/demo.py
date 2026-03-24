#!/usr/bin/env python3
"""
Open-Vocabulary 3D Scene Understanding - Interactive Gradio Demo
Enter a text query (e.g. "car . person") and upload an image.
The pipeline runs GroundingDINO → SAM 2 → Depth Anything V2 → 3D Lifting.
Outputs:
  - Side-by-side: Original | Grounded segmentation | Depth map
  - Bird's-eye view (BEV) of the 3D scene
  - Plotly 3D interactive point cloud
  - Table of detected objects with 3D positions

Usage:
    source activate.sh
    python app/demo.py --share      # to get a public URL
    python app/demo.py              # local only at port 7860
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))

# Some lab shells export proxy variables globally. Gradio imports httpx at module
# import time, and httpx can fail early if proxy extras are not installed. The demo
# is local-first and does not require those inherited proxy settings to start.
for _proxy_var in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_proxy_var, None)

import argparse, time, tempfile, json
from pathlib import Path
import numpy as np
import torch
import cv2
import gradio as gr
import plotly.graph_objects as go

from depth_anything_v2.dpt import DepthAnythingV2
from groundingdino.util.inference import load_model as load_gdino, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"
DEPTH_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
GDINO_CONFIG = ROOT / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

KITTI_K = np.array([
    [721.5377, 0.0,       609.5593],
    [0.0,      721.5377,  172.8540],
    [0.0,      0.0,       1.0     ]
], dtype=np.float32)

# Global model cache (loaded once)
_models = {}

COLORS_RGB = [
    (1.0, 0.25, 0.25), (0.25, 1.0, 0.25), (0.25, 0.25, 1.0),
    (1.0, 1.0, 0.25),  (1.0, 0.25, 1.0),  (0.25, 1.0, 1.0),
    (1.0, 0.6, 0.2),   (0.6, 0.2, 1.0),   (0.2, 1.0, 0.6),
]


def get_models():
    if _models:
        return _models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Demo] Loading models on {device}...")

    depth_model = DepthAnythingV2(**DEPTH_CONFIG)
    depth_model.load_state_dict(torch.load(WEIGHTS / "depth_anything_v2_vits.pth", map_location="cpu"))
    depth_model = depth_model.to(device).eval()

    gdino_model = load_gdino(str(GDINO_CONFIG), str(WEIGHTS / "groundingdino_swint_ogc.pth"), device=str(device))
    gdino_model.eval()

    sam2_model = build_sam2(SAM2_CONFIG, str(WEIGHTS / "sam2.1_hiera_small.pt"), device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)

    _models["depth"] = depth_model
    _models["gdino"] = gdino_model
    _models["sam2"] = predictor
    _models["device"] = device
    print("[Demo] All models ready.")
    return _models


def process(image_pil, text_query: str,
            box_thr: float, text_thr: float,
            max_depth: float, stride: int, use_kitti_k: bool):
    """Main inference function called by Gradio."""
    if image_pil is None:
        return None, None, None, "No image uploaded.", None

    t_start = time.perf_counter()
    models = get_models()
    device = models["device"]

    # Convert PIL → OpenCV BGR
    img_np = np.array(image_pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_rgb = img_np
    h, w = img_bgr.shape[:2]

    # Camera intrinsics
    if use_kitti_k:
        K = KITTI_K
    else:
        fx = w / (2 * np.tan(np.radians(70) / 2))
        K = np.array([[fx, 0, w/2], [0, fx, h/2], [0, 0, 1]], dtype=np.float32)

    # --- Depth ---
    t0 = time.perf_counter()
    with torch.no_grad():
        depth = models["depth"].infer_image(img_rgb, 518)
    t_depth = (time.perf_counter() - t0) * 1000

    d_min, d_max = depth.min(), depth.max()
    depth_metric = (depth - d_min) / (d_max - d_min + 1e-8) * max_depth + 0.5

    # --- GroundingDINO ---
    t0 = time.perf_counter()
    # Save temp image for load_image (it expects a file path)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, img_bgr)
        tmp_path = tmp.name
    img_src, img_tensor = load_image(tmp_path)
    os.unlink(tmp_path)

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=models["gdino"], image=img_tensor, caption=text_query,
            box_threshold=box_thr, text_threshold=text_thr, device=str(device)
        )
    t_gdino = (time.perf_counter() - t0) * 1000

    # Convert boxes
    boxes_xyxy = np.zeros((len(boxes), 4), dtype=np.float32)
    if len(boxes) > 0:
        b = boxes.cpu().numpy()
        boxes_xyxy[:, 0] = (b[:, 0] - b[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (b[:, 1] - b[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (b[:, 0] + b[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (b[:, 1] + b[:, 3] / 2) * h

    # --- SAM 2 ---
    t0 = time.perf_counter()
    masks = np.zeros((0, h, w), dtype=bool)
    if len(boxes_xyxy) > 0:
        models["sam2"].set_image(img_rgb)
        masks_raw, _, _ = models["sam2"].predict(
            point_coords=None, point_labels=None,
            box=boxes_xyxy.astype(np.float32),
            multimask_output=False,
        )
        if masks_raw.ndim == 4:
            masks_raw = masks_raw[:, 0]
        masks = masks_raw.astype(bool)
    t_sam2 = (time.perf_counter() - t0) * 1000

    t_total = (time.perf_counter() - t_start) * 1000

    # --- Visualization Panel ---
    panel = make_visualization_panel(img_bgr, depth, boxes_xyxy, logits, phrases, masks, t_depth, t_gdino, t_sam2)

    # --- BEV ---
    bev = make_bev(img_rgb, depth_metric, masks, phrases, K, max_depth, stride)

    # --- Plotly 3D ---
    fig_3d = make_plotly_3d(img_rgb, depth_metric, masks, phrases, K, max_depth, stride)

    # --- Report table ---
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    rows = []
    for i, (mask, phrase, logit) in enumerate(zip(masks, phrases, logits)):
        ys, xs = np.where(mask.astype(bool))
        if len(ys) == 0:
            continue
        zs = depth_metric[ys, xs]
        z_med = float(np.median(zs))
        x_med = float((np.median(xs) - cx) * z_med / fx)
        y_med = float((np.median(ys) - cy) * z_med / fy)
        rows.append({
            "Object": phrase,
            "Confidence": f"{float(logit):.2f}",
            "X (m)": f"{x_med:.1f}",
            "Y (m)": f"{y_med:.1f}",
            "Z depth (m)": f"{z_med:.1f}",
            "Pixels": len(ys),
        })
    report_md = f"**Query:** `{text_query}`  |  **Detections:** {len(boxes_xyxy)}  |  **Total:** {t_total:.0f}ms\n\n"
    report_md += f"Depth: {t_depth:.0f}ms | GDINO: {t_gdino:.0f}ms | SAM2: {t_sam2:.0f}ms\n\n"
    if rows:
        report_md += "| Object | Conf | X(m) | Y(m) | Z(m) | Pixels |\n"
        report_md += "|--------|------|------|------|------|--------|\n"
        for r in rows:
            report_md += f"| {r['Object']} | {r['Confidence']} | {r['X (m)']} | {r['Y (m)']} | {r['Z depth (m)']} | {r['Pixels']} |\n"
    else:
        report_md += "_No objects detected._"

    return panel, bev, fig_3d, report_md, gr.update(visible=True)


def make_visualization_panel(img_bgr, depth, boxes_xyxy, logits, phrases, masks, t_depth, t_gdino, t_sam2):
    h, w = img_bgr.shape[:2]
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_colored = cv2.resize(depth_colored, (w, h))

    seg_panel = img_bgr.copy()
    for i, (mask, phrase, logit) in enumerate(zip(masks, phrases, logits)):
        c = tuple(int(v * 255) for v in COLORS_RGB[i % len(COLORS_RGB)])[::-1]  # BGR
        overlay = seg_panel.copy()
        overlay[mask.astype(bool)] = c
        seg_panel = cv2.addWeighted(seg_panel, 0.55, overlay, 0.45, 0)
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cv2.rectangle(seg_panel, (x1, y1), (x2, y2), c, 2)
        cv2.putText(seg_panel, f"{phrase} {logit:.2f}", (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 2)

    gap = np.full((h, 3, 3), 40, dtype=np.uint8)
    result = np.hstack([img_bgr, gap, seg_panel, gap, depth_colored])
    info = f"GDINO:{t_gdino:.0f}ms  SAM2:{t_sam2:.0f}ms  Depth:{t_depth:.0f}ms  | {len(boxes_xyxy)} objects"
    cv2.putText(result, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def make_bev(img_rgb, depth_metric, masks, phrases, K, max_depth, stride):
    h, w = img_rgb.shape[:2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    img_size = 600

    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)
    dd = depth_metric[::stride, ::stride]
    valid = (dd > 0.1) & (dd < max_depth)

    X = ((uu - cx) * dd / fx)[valid]
    Z = dd[valid]
    colors_bev = (img_rgb[::stride, ::stride][valid] / 255.0)

    # Build BEV canvas
    bev = np.zeros((img_size, img_size, 3), dtype=np.float32)
    x_range = (X.min(), X.max())
    z_range = (Z.min(), Z.max())
    xi = ((X - x_range[0]) / (x_range[1] - x_range[0] + 1e-6) * (img_size - 1)).astype(int).clip(0, img_size - 1)
    zi = ((1 - (Z - z_range[0]) / (z_range[1] - z_range[0] + 1e-6)) * (img_size - 1)).astype(int).clip(0, img_size - 1)
    order = np.argsort(-Z)
    bev[zi[order], xi[order]] = colors_bev[order]

    # Overlay object centroids on BEV
    bev_img = (bev * 255).astype(np.uint8)
    for i, (mask, phrase) in enumerate(zip(masks, phrases)):
        ys, xs = np.where(mask.astype(bool))
        if len(ys) == 0:
            continue
        zs = depth_metric[ys, xs]
        z_med = float(np.median(zs))
        x_med = float((np.median(xs) - cx) * z_med / fx)
        bx = int((x_med - x_range[0]) / (x_range[1] - x_range[0] + 1e-6) * (img_size - 1))
        bz = int((1 - (z_med - z_range[0]) / (z_range[1] - z_range[0] + 1e-6)) * (img_size - 1))
        c = tuple(int(v * 255) for v in COLORS_RGB[i % len(COLORS_RGB)])
        cv2.circle(bev_img, (bx, bz), 8, c, -1)
        cv2.putText(bev_img, f"{phrase[:3]} {z_med:.0f}m", (bx + 10, bz),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)
    cv2.putText(bev_img, "Bird's-Eye View (top-down, Z=depth, X=lateral)",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return bev_img


def make_plotly_3d(img_rgb, depth_metric, masks, phrases, K, max_depth, stride):
    h, w = img_rgb.shape[:2]
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    u = np.arange(0, w, stride * 2)  # sparser for Plotly
    v = np.arange(0, h, stride * 2)
    uu, vv = np.meshgrid(u, v)
    dd = depth_metric[::stride*2, ::stride*2]
    valid = (dd > 0.1) & (dd < max_depth)

    X = ((uu - cx) * dd / fx)[valid]
    Y = -((vv - cy) * dd / fy)[valid]  # flip Y for standard 3D view
    Z = dd[valid]
    colors_base = img_rgb[::stride*2, ::stride*2][valid]

    # Base scatter (grey background)
    traces = [go.Scatter3d(
        x=X, y=Z, z=Y,
        mode='markers',
        marker=dict(
            size=1.5,
            color=[f'rgb({r},{g},{b})' for r, g, b in colors_base],
            opacity=0.4,
        ),
        name='Scene',
        hoverinfo='skip',
    )]

    # Object-specific traces (vivid colors, larger dots)
    for i, (mask, phrase) in enumerate(zip(masks, phrases)):
        ys, xs = np.where(mask.astype(bool))
        if len(ys) == 0:
            continue
        # Subsample for Plotly performance
        idx = np.random.choice(len(ys), min(len(ys), 2000), replace=False)
        ys_s, xs_s = ys[idx], xs[idx]
        zs = depth_metric[ys_s, xs_s]
        Xo = (xs_s - cx) * zs / fx
        Yo = -((ys_s - cy) * zs / fy)
        Zo = zs
        r, g, b = (int(c * 255) for c in COLORS_RGB[i % len(COLORS_RGB)])
        traces.append(go.Scatter3d(
            x=Xo, y=Zo, z=Yo,
            mode='markers',
            marker=dict(size=3, color=f'rgb({r},{g},{b})', opacity=0.9),
            name=f"{phrase} (#{i+1})",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis_title="X (m) lateral",
            yaxis_title="Z (m) depth",
            zaxis_title="Y (m) vertical",
            bgcolor="rgb(15,15,25)",
            xaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgb(40,40,60)"),
            yaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgb(40,40,60)"),
            zaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="rgb(40,40,60)"),
        ),
        paper_bgcolor="rgb(15,15,25)",
        plot_bgcolor="rgb(15,15,25)",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(30,30,50,0.8)", font=dict(size=10)),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text="Interactive 3D Scene Point Cloud", font=dict(size=16)),
        height=500,
    )
    return fig


# ---- CSS ----
CSS = """
body { background-color: #f8fafc; }
.gradio-container { max-width: 1400px !important; padding-top: 20px !important; }
#title-container { 
    background: linear-gradient(135deg, #4f46e5 0%, #ec4899 100%); 
    padding: 30px; 
    border-radius: 16px; 
    color: white; 
    text-align: center; 
    margin-bottom: 25px; 
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
}
#title-container h1 { color: white; margin: 0; font-weight: 800; font-size: 2.8em; letter-spacing: -0.02em; }
#title-container p { color: #f8fafc; font-size: 1.15em; margin-top: 12px; font-weight: 300; }
#title-container code { background: rgba(255,255,255,0.2); color: white; border-radius: 4px; padding: 2px 6px; }
.panel-box { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.02); border: 1px solid #e2e8f0; }
#run-btn { 
    background: linear-gradient(135deg, #6366f1, #8b5cf6); 
    color: white; 
    font-size: 18px; 
    font-weight: 600; 
    border-radius: 8px; 
    transition: all 0.2s ease;
    border: none;
    padding: 12px;
}
#run-btn:hover { 
    transform: translateY(-2px); 
    box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4); 
}
"""

# ---- Gradio UI ----
def build_demo():
    examples = [
        [str(p), "car . person . truck", 0.35, 0.25, 50.0, 2]
        for p in sorted((ROOT / "data" / "demo_inputs").glob("*.png"))[:3]
    ]

    theme = gr.themes.Soft(
        primary_hue="indigo", 
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )

    with gr.Blocks(theme=theme, head=f"<style>{CSS}</style>", title="Open-Vocabulary 3D Scene Understanding") as demo:
        gr.HTML('''
        <div id="title-container">
            <h1>🔭 Open-Vocabulary 3D Scene Understanding</h1>
            <p><strong>Pipeline:</strong> Grounding DINO &rarr; SAM 2 &rarr; Depth Anything V2 &rarr; 3D Lifting</p>
            <p>Type any text query (e.g. <code>car . person . traffic light</code>) to detect, segment, and locate objects in 3D.</p>
        </div>
        ''')
        
        with gr.Row():
            with gr.Column(scale=4, elem_classes="panel-box"):
                gr.Markdown("### 📥 Input Setup")
                image_in = gr.Image(label="Input Image", type="pil", height=400)
                text_in = gr.Textbox(label="Text Query (labels separated by ' . ')",
                                     value="car . person . truck",
                                     placeholder="e.g. car . person . traffic light")
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    with gr.Row():
                        box_thr = gr.Slider(0.1, 0.9, 0.35, step=0.05, label="Box Threshold")
                        text_thr = gr.Slider(0.1, 0.9, 0.25, step=0.05, label="Text Threshold")
                    with gr.Row():
                        max_depth = gr.Slider(5, 100, 50, step=5, label="Max Depth (m)")
                        stride = gr.Slider(1, 8, 2, step=1, label="Point Cloud Stride")
                    kitti_k = gr.Checkbox(label="Use KITTI camera intrinsics", value=True)
                run_btn = gr.Button("🚀 Run Pipeline", elem_id="run-btn")

            with gr.Column(scale=6, elem_classes="panel-box"):
                with gr.Tabs():
                    with gr.TabItem("🖼️ 2D & Bird's-Eye View"):
                        panel_out = gr.Image(label="Pipeline: Original | Segmentation | Depth", height=300)
                        bev_out = gr.Image(label="Bird's-Eye View (BEV)", height=480)
                    with gr.TabItem("📊 Analysis Report"):
                        report_out = gr.Markdown("_Run the pipeline to see results._")
                        
        with gr.Row():
            with gr.Column(elem_classes="panel-box"):
                gr.Markdown("### 🧊 Interactive 3D Point Cloud")
                plotly_out = gr.Plot(label="Interactive 3D Point Cloud", visible=False)

        if examples:
            with gr.Row():
                with gr.Column(elem_classes="panel-box"):
                    gr.Examples(examples=examples,
                                inputs=[image_in, text_in, box_thr, text_thr, max_depth, stride],
                                label="Gallery of KITTI Examples")

        run_btn.click(
            fn=process,
            inputs=[image_in, text_in, box_thr, text_thr, max_depth, stride, kitti_k],
            outputs=[panel_out, bev_out, plotly_out, report_out, plotly_out],
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Enable Gradio public share link")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Pre-load models
    get_models()

    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
    )
