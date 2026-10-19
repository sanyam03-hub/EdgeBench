# EdgeBench report — resnet18 on cpu_1t

Run ID: `20261019T142558Z_7b9d053d` · created 2026-10-19T14:25:58Z · EdgeBench 0.1.0

## Results

| Variant | Status | Top-1 (%) | Δ vs FP32 (pts) | Top-1 agreement vs torch | Max abs err | p50 ms (b=1) | p90 ms (b=1) | CV (%) | p50 ms (b=8) | Size (MB) | Load (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_fp32 | ok | 97.70 | — | — | — | 58.25 | 67.51 | 13.7 (noisy) | 349.71 | — | 278.6 |
| onnx_fp32 | ok | 97.70 | +0.00 | 1.000 | 1.7e-05 | 36.19 | 41.08 | 10.1 (noisy) | 301.04 | 46.75 | 228.9 |
| onnx_int8_dynamic | ok | 97.60 | -0.10 | 0.969 | 1.5e+00 | 552.45 | 561.22 | 1.1 | 4339.73 | 11.74 | 111.3 |
| onnx_int8_static | ok | 96.20 | -1.50 | 1.000 | 1.2e+01 | 21.21 | 25.97 | 16.3 (noisy) | 157.51 | 11.82 | 221.6 |

Δ vs FP32 is measured against `onnx_fp32` (same runtime). Top-1 agreement and max abs err compare each variant with the PyTorch FP32 reference on the parity batch. `(noisy)`: CV above the target's threshold.

## Environment

- CPU: AMD Ryzen 5 5600H with Radeon Graphics (12 logical cores)
- OS: Windows 10.0.26200
- Python 3.11.4; numpy 2.4.6, onnx 1.23.2, onnxruntime 1.31.0, torch 2.14.1+cpu, torchvision 0.29.1+cpu
- Threads: intra-op 1, inter-op 1
- Batch sizes: [1, 8]; warmup 20, timed runs 200

## Methodology

- Accuracy: 10-way top-1 among the Imagenette classes (argmax over their 10 logits of the 1000-class output) on 1000 images from `val/`. Not ImageNet top-1; do not compare with published ImageNet numbers. Meaningless for the random `tiny_cnn` smoke model.
- Calibration (INT8 static): 256 images from `train/` only, disjoint from eval.
- Timing: `time.perf_counter_ns` around the inference call only; 20 warmup runs discarded, 200 timed runs recorded; p50/p90 by linear interpolation; CV = std/mean.
- Gated (`edgebench gate`): presence, status, top-1 drop vs baseline, parity. Latency is reported, never gated.

## Limitations

- CPU only (PyTorch eager and ONNX Runtime CPUExecutionProvider) on one machine; not representative of NPU/DSP or on-device mobile performance.
- Latency depends on the CPU, thread count and background load; compare runs only within the same target profile and machine.
