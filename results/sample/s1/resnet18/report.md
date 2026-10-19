# S1 — per-layer INT8 error, resnet18

Created 2026-10-19T17:22:34Z · target `cpu_4t` · AMD Ryzen 5 5600H with Radeon Graphics · onnxruntime 1.31.0

Static QDQ INT8 (calibration: 256 train/ images). Eval subset split (seed 0) into 500 search / 500 report images. Rankings on 32 search images; 32 quantized nodes can be kept in FP32.

## Top 10 by worst local SQNR (lower SQNR = keep in FP32 first)

| Rank | Node | Op | SQNR (dB) | Source |
|---|---|---|---|---|
| 1 | `layer2/layer2.0/Add` | Add | 28.2 | activation |
| 2 | `layer3/layer3.1/Add` | Add | 29.5 | activation |
| 3 | `layer3/layer3.0/Add` | Add | 29.7 | activation |
| 4 | `layer1/layer1.1/Add` | Add | 29.8 | activation |
| 5 | `layer1/layer1.0/Add` | Add | 30.5 | activation |
| 6 | `layer2/layer2.1/Add` | Add | 31.0 | activation |
| 7 | `layer4/layer4.0/Add` | Add | 31.6 | activation |
| 8 | `layer4/layer4.1/Add` | Add | 32.6 | activation |
| 9 | `layer4.0/conv1/Conv` | Conv | 33.4 | activation |
| 10 | `layer4.0/downsample.0/Conv` | Conv | 33.4 | activation |

## Top 10 by most sensitive (logit SQNR) (lower SQNR = keep in FP32 first)

| Rank | Node | Op | SQNR (dB) | Source |
|---|---|---|---|---|
| 1 | `conv1/Conv` | Conv | 10.4 | logits |
| 2 | `layer1.0/conv1/Conv` | Conv | 28.9 | logits |
| 3 | `layer1/layer1.0/Add` | Add | 29.9 | logits |
| 4 | `layer3/layer3.1/Add` | Add | 33.7 | logits |
| 5 | `fc/Gemm` | Gemm | 34.4 | logits |
| 6 | `layer1.1/conv1/Conv` | Conv | 34.9 | logits |
| 7 | `layer3/layer3.0/Add` | Add | 34.9 | logits |
| 8 | `layer1.0/conv2/Conv` | Conv | 35.4 | logits |
| 9 | `layer4/layer4.0/Add` | Add | 35.7 | logits |
| 10 | `layer2/layer2.0/Add` | Add | 36.0 | logits |
