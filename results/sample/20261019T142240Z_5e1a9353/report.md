# EdgeBench report — mobilenet_v3_small on cpu_1t

Run ID: `20261019T142240Z_5e1a9353` · created 2026-10-19T14:22:40Z · EdgeBench 0.1.0

## Results

| Variant | Status | Top-1 (%) | Δ vs FP32 (pts) | Top-1 agreement vs torch | Max abs err | p50 ms (b=1) | p90 ms (b=1) | CV (%) | p50 ms (b=8) | Size (MB) | Load (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| torch_fp32 | ok | 97.60 | — | — | — | 7.01 | 8.56 | 15.1 (noisy) | 44.05 | — | 230.7 |
| onnx_fp32 | ok | 97.60 | +0.00 | 1.000 | 6.5e-05 | 2.48 | 3.85 | 30.5 (noisy) | 19.52 | 10.18 | 74.8 |
| onnx_int8_dynamic | ok | 14.10 | -83.50 | 0.000 | 2.9e+01 | 32.85 | 40.48 | 12.4 (noisy) | 270.99 | 2.72 | 118.8 |
| onnx_int8_static | ok | 25.20 | -72.40 | 0.000 | 2.7e+01 | 3.35 | 4.18 | 14.6 (noisy) | 24.69 | 2.91 | 137.8 |

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
