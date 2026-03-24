# 项目2：3D视觉感知 Pipeline（深度估计 + 手部/姿态追踪 + TensorRT 加速）

**可行性：4.5/5 - T4 的 INT8 加速能力让这个项目极其适合**

> JD 覆盖：深度估计、立体匹配、手部追踪、身体姿态估计、TensorRT/CUDA 加速、视觉模型部署优化

| 维度 | 情况 |
|------|------|
| 数据 | KITTI（~12GB）、NYU Depth V2、InterHand2.6M、COCO-WholeBody，4.4TB 磁盘充裕 |
| 代码 | Depth Anything V2（深度估计）、HaMeR/100DOH（手部追踪）、MMPose（姿态估计）均有完整开源代码 |
| GPU 适配 | T4 是推理优化的最佳练手卡：INT8 = 130 TOPS，TensorRT 对 T4 优化最成熟 |
| 双卡利用 | 一张跑模型推理基准，一张跑 TensorRT 优化版，直接对比加速效果 |
| 耗时 | 3-4 周 |
| 核心亮点 | T4 本身就是推理卡，用它做 TensorRT 加速实验是最合理的硬件搭配 |

## 做法建议

1. **深度估计模块：** 部署 Depth Anything V2（ViT-S/B），在 KITTI 上评测精度，然后用 TensorRT FP16 / INT8 量化加速
2. **手部/姿态追踪模块：** HaMeR（3D 手部重建）或 MMPose（全身姿态），完成推理 + 可视化
3. **TensorRT 加速（重点）：** 对每个模块做 PyTorch -> ONNX -> TensorRT 转换，对比 FP32 / FP16 / INT8 的推理速度和精度损失，输出详细 benchmark 报告
4. **集成 Pipeline：** 将多模块整合为统一感知 Pipeline，测量端到端延迟

> T4 本身就是 NVIDIA 为推理场景设计的，用它来展示 TensorRT 优化能力非常有说服力。

## 可展示成果（面试必备）

| 产出物 | 具体内容 | 展示方式 |
|--------|----------|----------|
| **可视化 Demo** | 输入一张 RGB 图，同时输出深度图、3D 手部网格、全身骨架叠加的可视化结果 | 拼接对比图/视频，非常直观 |
| **Benchmark 表格** | 每个模块在 FP32 / FP16 / INT8 下的推理延迟（ms）、吞吐量（FPS）、精度指标（如 AbsRel/RMSE） | Markdown 表格，这是最硬核的展示 |
| **加速比图表** | TensorRT 优化前后的速度柱状图（如 “PyTorch FP32: 45ms -> TensorRT INT8: 9ms”，加速 5x） | 柱状图放 README 最醒目位置 |
| **端到端 Pipeline Demo** | 视频输入，实时跑完深度估计 + 手部追踪 + 姿态估计的全流程，显示总延迟和每模块耗时 | 录屏视频 |
| **GitHub 仓库** | 含 `export_onnx.py`、`trt_inference.py`、`benchmark.py`、`visualize.py` 等完整工具链 | 代码结构清晰，README 放结果表格和可视化图 |

---

## Context

**目标：** 在 `/data/ganyw/3D` 下完成一个可部署、可 benchmark、可写进简历的 3D 视觉感知项目，包含：
- 单目深度估计
- 人体姿态或手部关键点 / 3D 手部重建
- PyTorch -> ONNX -> TensorRT 的推理优化链路
- 端到端延迟、吞吐、精度损失的定量报告

**当前硬件与环境：**
- GPU: 2 x Tesla T4（各 15GB VRAM）
- Driver: 570.181
- Python: 3.10.13
- PyTorch: 2.5.1 + CUDA 12.4
- 工作目录：`/data/ganyw/3D`

**结论先行：**
- 这台机器更适合做“推理优化型项目”，而不是继续做 OpenPCDet 训练
- T4 对 FP16 / INT8 TensorRT 优化很合适，项目价值比纯训练复现更高
- 当前环境还没装 ONNX / ONNX Runtime / TensorRT，需要先补部署链路

**最终产出物：**
1. 深度估计模块：PyTorch / ONNX / TensorRT 三套推理结果与 benchmark
2. 姿态或手部模块：推理脚本、可视化结果、性能对比
3. 统一 Pipeline：输入图片或视频，输出深度图 + 姿态 / 手部结果
4. benchmark 报告：FP32 / FP16 / INT8 延迟、吞吐、显存、精度变化
5. 简历描述与项目总结

---

## 项目范围与取舍

### 推荐最小可行版本

先做两模块，不要一开始就把范围铺太大：

1. **深度估计：Depth Anything V2**
2. **姿态估计：MMPose**

原因：
- 两者都容易形成“精度 + 部署优化 + 可视化”的完整闭环
- 比 HaMeR 更稳，环境依赖更可控
- 先做通 PyTorch -> ONNX -> TensorRT 流程，再决定是否追加手部重建

### 第三模块是否加入 HaMeR

可做，但建议放到第二阶段：
- HaMeR 的展示效果强
- 但依赖更重，转换 ONNX / TensorRT 的坑更多
- 适合在前两个模块跑通之后作为加分项

---

## 目录结构规划

```text
/data/ganyw/3D/
├── .venv/
├── data/
│   ├── kitti/                      # 深度评测可用
│   ├── nyu_depth_v2/               # 可选
│   ├── coco_wholebody/             # 姿态估计可选
│   └── demo_inputs/                # 测试图片/视频
├── models/
│   ├── depth_anything_v2/
│   ├── mmpose/
│   ├── hamer/                      # 第二阶段可选
│   └── weights/
├── exports/
│   ├── onnx/
│   └── tensorrt/
├── benchmarks/
│   ├── depth/
│   ├── pose/
│   └── pipeline/
├── visualizations/
│   ├── depth/
│   ├── pose/
│   └── pipeline/
├── scripts/
│   ├── export_onnx/
│   ├── build_trt/
│   ├── benchmark/
│   └── run_pipeline/
└── KITTI_3D_检测项目.md            # 本计划文件
```

---

## 技术路线

### 模块 A：深度估计

**首选：Depth Anything V2**

目标：
- 跑通预训练模型推理
- 在 KITTI 或 demo 图像上输出深度图
- 导出 ONNX
- 构建 TensorRT engine
- 对比 PyTorch / ONNX Runtime / TensorRT 的速度与精度

建议模型：
- `ViT-S` 先跑通
- `ViT-B` 作为更高精度对照组

### 模块 B：姿态估计

**首选：MMPose**

目标：
- 跑通 2D 全身姿态估计
- 输出关键点可视化
- 导出 ONNX
- 做 TensorRT FP16 优化

建议先做：
- 先做人全身 2D pose
- 如果顺利，再加 hand keypoints 或 wholebody

### 模块 C：手部重建

**可选：HaMeR**

目标：
- 跑通单张图/视频的 3D 手部重建
- 完成可视化
- 评估其 ONNX / TensorRT 可转换性

备注：
- 这个模块更适合作为“高阶加分项”
- 如果时间有限，可以只写为第二阶段扩展计划

---

## Step 1：环境检查与依赖补齐

### 当前已知状态

当前 `/data/ganyw/3D` 环境里：
- PyTorch 可用
- GPU 可见（2 x T4）
- 但尚未安装：
  - `onnx`
  - `onnxruntime` / `onnxruntime-gpu`
  - `tensorrt`
  - `polygraphy`
- 当前也没有 `trtexec`

### 目标依赖

```bash
cd /data/ganyw/3D
source .venv/bin/activate

pip install onnx onnxruntime-gpu polygraphy
pip install opencv-python pillow matplotlib tqdm pyyaml
```

TensorRT 部分分两种情况：

1. **系统已安装 TensorRT**
   - 验证 `trtexec --version`
   - 安装 Python 绑定后直接使用

2. **系统未安装 TensorRT**
   - 需要单独安装 TensorRT SDK 或 Python wheel
   - 然后验证 `import tensorrt`

### 验证命令

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import onnx; print(onnx.__version__)"
python -c "import onnxruntime as ort; print(ort.__version__)"
python -c "import tensorrt as trt; print(trt.__version__)"
trtexec --version
```

---

## Step 2：拉取项目代码

### Depth Anything V2

```bash
cd /data/ganyw/3D/models
git clone https://github.com/DepthAnything/Depth-Anything-V2.git depth_anything_v2
```

### MMPose

```bash
cd /data/ganyw/3D/models
git clone https://github.com/open-mmlab/mmpose.git
```

### HaMeR（第二阶段可选）

```bash
cd /data/ganyw/3D/models
git clone https://github.com/geopavlakos/hamer.git
```

---

## Step 3：准备数据与输入样本

### 深度估计

优先级建议：

1. **先用 demo 图片快速跑通**
2. **再接 KITTI 做定量评测**
3. **如果需要室内场景，再补 NYU Depth V2**

KITTI 可用于：
- 深度图可视化
- 若模型支持评测，可做误差指标统计
- 展示户外自动驾驶场景适配性

### 姿态估计

先不急着完整下载 COCO-WholeBody，建议先：
- 用 demo 图像 / 视频跑通可视化
- 跑通导出与 TensorRT
- 需要定量指标时再决定是否补完整数据集

### demo_inputs 建议

在 `/data/ganyw/3D/data/demo_inputs/` 准备：
- 10-20 张室内外图片
- 1-2 段短视频
- 包含单人、多人、手部明显的场景

---

## Step 4：先跑通 PyTorch 推理

### 深度估计模块目标

最低要求：
- 输入 RGB 图像
- 输出深度图 `.png`
- 记录单张图平均推理时间

输出目录建议：

```text
/data/ganyw/3D/visualizations/depth/
```

### 姿态估计模块目标

最低要求：
- 输入图像或视频
- 输出关键点可视化图/视频
- 记录平均推理延迟

输出目录建议：

```text
/data/ganyw/3D/visualizations/pose/
```

### 这一阶段的完成标准

- 两个模块都能在 PyTorch 下稳定推理
- 能保存结果文件
- 能统计基本性能数据

---

## Step 5：导出 ONNX

### 深度估计导出目标

导出到：

```text
/data/ganyw/3D/exports/onnx/depth_anything_v2_*.onnx
```

检查项：
- 输入输出 shape 明确
- 动态 batch 是否需要支持
- 动态分辨率是否需要支持
- ONNX 和 PyTorch 输出误差是否在可接受范围内

### 姿态估计导出目标

导出到：

```text
/data/ganyw/3D/exports/onnx/mmpose_*.onnx
```

检查项：
- 后处理是否在模型图内
- 是否需要拆分 pre/post-process
- 是否存在 TensorRT 不支持的算子

### ONNX 验证

```bash
python -c "import onnx; m=onnx.load('xxx.onnx'); onnx.checker.check_model(m); print('onnx ok')"
```

---

## Step 6：TensorRT 构建与优化

### 优先顺序

先做：
1. FP32 baseline
2. FP16 engine
3. INT8 engine（如果校准数据和算子支持足够稳定）

### 构建产物

```text
/data/ganyw/3D/exports/tensorrt/
├── depth_anything_v2_fp16.engine
├── depth_anything_v2_int8.engine
├── mmpose_fp16.engine
└── mmpose_int8.engine
```

### benchmark 维度

每个模块都要对比：
- PyTorch FP32
- ONNX Runtime
- TensorRT FP16
- TensorRT INT8

统计指标：
- 平均延迟（ms）
- P50 / P95 延迟
- FPS
- 显存占用
- 与 PyTorch baseline 的输出误差

### T4 上的预期价值

简历里真正有说服力的是：
- 同一模型在 T4 上完成从 PyTorch 到 TensorRT 的部署优化
- FP16/INT8 带来明确加速比
- 加速后精度损失可量化、可解释

---

## Step 7：集成统一 Pipeline

### Pipeline 输入

- 单张图片
- 文件夹批量图片
- 短视频

### Pipeline 输出

- 深度图
- 姿态可视化图
- 统一叠加结果
- 每帧耗时与总耗时日志

### 推荐结构

```text
scripts/run_pipeline/
├── run_depth.py
├── run_pose.py
├── run_pipeline.py
└── utils.py
```

### 端到端评测

重点不是只看单模型速度，而是：
- 单模块延迟
- 串联后总延迟
- I/O 与预处理后处理占比
- 是否达到“接近实时”或“离线高吞吐”目标

---

## Step 8：结果汇总与 benchmark 报告

建议在 `/data/ganyw/3D/benchmarks/` 下保存：

```text
depth/summary.md
pose/summary.md
pipeline/summary.md
```

报告至少包含：
- 模型版本
- 输入分辨率
- batch size
- GPU 型号
- PyTorch / ONNX / TensorRT 的延迟与 FPS
- FP16 / INT8 相对 FP32 的误差变化
- 典型可视化结果

### 建议展示表格

| 模块 | 后端 | 精度 | 平均延迟 | FPS | 备注 |
|------|------|------|----------|-----|------|
| Depth Anything V2 | PyTorch | FP32 | [填入] | [填入] | baseline |
| Depth Anything V2 | TensorRT | FP16 | [填入] | [填入] | 主推配置 |
| Depth Anything V2 | TensorRT | INT8 | [填入] | [填入] | 需校准 |
| MMPose | PyTorch | FP32 | [填入] | [填入] | baseline |
| MMPose | TensorRT | FP16 | [填入] | [填入] | 主推配置 |

---

## 简历价值应如何表述

这类项目的卖点不是“我复现了某个模型”，而是：
- 你能搭建完整视觉推理链路
- 你会做模型部署优化
- 你会做 benchmark 和工程化取舍
- 你知道 T4 这类推理卡怎么发挥价值

### 中文简历描述方向

- 搭建基于 Tesla T4 的 3D 视觉感知 Pipeline，集成单目深度估计与人体姿态估计模块，完成 PyTorch、ONNX Runtime 与 TensorRT 多后端部署
- 实现 FP16 / INT8 推理优化与 benchmark，对比端到端延迟、吞吐与精度损失，形成可复现实验报告

### 英文简历描述方向

- Built a 3D perception pipeline on dual Tesla T4 GPUs, integrating monocular depth estimation and pose estimation with multi-backend deployment across PyTorch, ONNX Runtime, and TensorRT
- Optimized inference with FP16/INT8 TensorRT engines and delivered reproducible latency, throughput, and accuracy benchmarks for end-to-end perception workloads

---

## 推荐执行顺序

### 第一周

1. 补齐 ONNX / TensorRT 环境
2. 跑通 Depth Anything V2 PyTorch 推理
3. 导出 ONNX 并验证
4. 做 TensorRT FP16 benchmark

### 第二周

1. 跑通 MMPose PyTorch 推理
2. 导出 ONNX
3. 做 TensorRT FP16 benchmark
4. 汇总两个模块的报告

### 第三周

1. 组装统一 Pipeline
2. 跑端到端延迟测试
3. 视情况增加 INT8 或 HaMeR
4. 输出最终项目总结与简历描述

---

## 当前真实状态

截至现在：
- `/data/ganyw/3D` 下 GPU 可用
- PyTorch 可用
- 这台机器适合做项目2
- 但部署依赖尚未装齐，当前还不能直接开始 TensorRT benchmark

**下一步最合理的动作：**

1. 安装 `onnx`、`onnxruntime-gpu`
2. 确认 `TensorRT / trtexec` 的可用方式
3. 先拉 `Depth Anything V2`
4. 先完成第一个可跑通的深度估计 baseline

---

## 关键参考

- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2
- MMPose: https://github.com/open-mmlab/mmpose
- HaMeR: https://github.com/geopavlakos/hamer
- TensorRT: https://github.com/NVIDIA/TensorRT
- ONNX Runtime: https://github.com/microsoft/onnxruntime
- torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
