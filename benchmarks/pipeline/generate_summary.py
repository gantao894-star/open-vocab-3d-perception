#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/data/ganyw/3D')
SUMMARY = ROOT / 'benchmarks' / 'pipeline' / 'summary.md'

SECTIONS = {
    'Depth Anything V2 (ViT-S, 518x518)': [
        ROOT / 'benchmarks' / 'depth' / 'pytorch_fp32_vits_518_cuda.json',
        ROOT / 'benchmarks' / 'depth' / 'onnxruntime_cuda_vits_518.json',
        ROOT / 'benchmarks' / 'depth' / 'onnxruntime_TensorrtExecutionProvider_vits_518.json',
        ROOT / 'benchmarks' / 'depth' / 'tensorrt_fp16_vits_518.json',
        ROOT / 'benchmarks' / 'depth' / 'onnxruntime_trt_ep_vits_518.json',
    ],
    'MMPose 2D Pose': [
        ROOT / 'benchmarks' / 'mmpose' / 'pytorch_fp32.json',
        ROOT / 'benchmarks' / 'mmpose' / 'onnxruntime_cuda.json',
        ROOT / 'benchmarks' / 'mmpose' / 'onnxruntime_trt.json',
    ],
}


def load_first(paths):
    for path in paths:
        if path.exists():
            return path, json.loads(path.read_text())
    return None, None


def collect_rows(paths):
    rows = []
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        rows.append((path, data))
    return rows


lines = [
    '# Benchmark Summary',
    '',
    'Generated from JSON artifacts under `benchmarks/`.',
    '',
]

for title, paths in SECTIONS.items():
    lines.append(f'## {title}')
    rows = collect_rows(paths)
    if not rows:
        lines.append('No benchmark artifacts found.')
        lines.append('')
        continue

    lines.append('| Backend | Status | Mean ms | FPS | P95 ms | Notes | Source |')
    lines.append('|---|---:|---:|---:|---:|---|---|')
    for path, data in rows:
        status = data.get('status', 'ok')
        metrics = data.get('metrics', {})
        mean_ms = f"{metrics.get('mean_ms', 0.0):.2f}" if metrics else '-'
        fps = f"{metrics.get('fps', 0.0):.2f}" if metrics else '-'
        p95 = f"{metrics.get('p95_ms', 0.0):.2f}" if metrics else '-'
        notes = []
        if data.get('providers_active'):
            notes.append('active=' + ','.join(data['providers_active']))
        if data.get('cache_artifacts'):
            notes.append(f"cache={len(data['cache_artifacts'])} files")
        if status != 'ok' and data.get('error'):
            notes.append(data['error'].splitlines()[0][:120])
        lines.append(
            f"| {data.get('backend', path.stem)} | {status} | {mean_ms} | {fps} | {p95} | {'; '.join(notes) or '-'} | `{path.relative_to(ROOT)}` |"
        )
    lines.append('')

lines.extend([
    '## Blockers',
    '- TensorRT engine generation on this machine is currently blocked if `libnvinfer.so.10` is missing from the runtime library path.',
    '- ONNX Runtime may advertise `TensorrtExecutionProvider`, but provider activation still fails until TensorRT runtime libraries are installed and discoverable.',
    '',
])

SUMMARY.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(SUMMARY)
