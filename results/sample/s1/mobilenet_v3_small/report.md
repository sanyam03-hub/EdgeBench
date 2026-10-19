# S1 — per-layer INT8 error, mobilenet_v3_small

Created 2026-10-19T17:08:40Z · target `cpu_4t` · AMD Ryzen 5 5600H with Radeon Graphics · onnxruntime 1.31.0

Static QDQ INT8 (calibration: 256 train/ images). Eval subset split (seed 0) into 500 search / 500 report images. Rankings on 32 search images; 127 quantized nodes can be kept in FP32.

## Top 10 by worst local SQNR (lower SQNR = keep in FP32 first)

| Rank | Node | Op | SQNR (dB) | Source |
|---|---|---|---|---|
| 1 | `features.11/block.3.0/Conv` | Conv | 15.3 | activation |
| 2 | `features.10/block.3.0/Conv` | Conv | 20.0 | activation |
| 3 | `features.11/block.2/Mul` | Mul | 21.2 | activation |
| 4 | `features.11/avgpool/GlobalAveragePool` | GlobalAveragePool | 21.2 | activation |
| 5 | `features/features.11/Add` | Add | 24.0 | activation |
| 6 | `avgpool/GlobalAveragePool` | GlobalAveragePool | 24.0 | activation |
| 7 | `HardSigmoid:_inlfunc_HardSwish_token_15_HS_X` | HardSigmoid | 24.3 | activation |
| 8 | `Mul:features.11/block.1.2/HardSwish_output_0` | Mul | 24.3 | activation |
| 9 | `features.8/block.2/Mul` | Mul | 25.8 | activation |
| 10 | `features.8/avgpool/GlobalAveragePool` | GlobalAveragePool | 25.8 | activation |

## Top 10 by most sensitive (logit SQNR) (lower SQNR = keep in FP32 first)

| Rank | Node | Op | SQNR (dB) | Source |
|---|---|---|---|---|
| 1 | `features.0/features.0.0/Conv` | Conv | -0.1 | logits |
| 2 | `features.1/block.0.0/Conv` | Conv | 0.1 | logits |
| 3 | `Mul:features.0/features.0.2/HardSwish_output_0` | Mul | 0.1 | logits |
| 4 | `features.3/block.0.0/Conv` | Conv | 5.1 | logits |
| 5 | `features.2/block.0.0/Conv` | Conv | 14.0 | logits |
| 6 | `features.2/block.1.0/Conv` | Conv | 19.0 | logits |
| 7 | `features.1/block.2.0/Conv` | Conv | 23.2 | logits |
| 8 | `Mul:features.11/block.1.2/HardSwish_output_0` | Mul | 23.4 | logits |
| 9 | `Mul:features.7/block.0.2/HardSwish_output_0` | Mul | 23.4 | logits |
| 10 | `features.4/block.1.0/Conv` | Conv | 23.6 | logits |

## Search half: keep the top-k nodes in FP32 (FP32 top-1 98.00%)

| k | worst local SQNR (%) | most sensitive (logit SQNR) (%) | random k nodes (%), per seed |
|---|---|---|---|
| 0 | 26.40 | 26.40 | — |
| 1 | 26.40 | 27.00 | 24.60, 22.20, 24.20 |
| 16 | 24.40 | 97.40 | 26.00, 26.40, 24.60 |
| 2 | 26.00 | 35.80 | 24.80, 23.40, 25.40 |
| 32 | 24.40 | 97.20 | 20.40, 23.60, 95.20 |
| 4 | 24.80 | 95.80 | 24.80, 24.20, 25.60 |
| 8 | 25.80 | 96.20 | 23.80, 31.40, 23.00 |

Chosen k (smallest k within 1.0 pts of FP32 on the search half, else the best): worst local SQNR k = 0, most sensitive (logit SQNR) k = 16.

## Report half (not used for any choice)

| Model | Top-1 (%) | Size (MB) | p50 ms (b=1) | p90 ms (b=1) | CV (%) |
|---|---|---|---|---|---|
| ONNX FP32 | 97.20 | 10.18 | 1.25 | 1.72 | 22.4 (noisy) |
| INT8 static, all nodes | 24.00 | 2.91 | 1.88 | 2.13 | 8.6 |
| INT8, top 0 by worst local SQNR in FP32 | 24.00 | 2.91 | 1.88 | 2.41 | 11.6 (noisy) |
| INT8, top 16 by most sensitive (logit SQNR) in FP32 | 97.00 | 2.93 | 1.60 | 2.02 | 11.5 (noisy) |

Random baseline for worst local SQNR (same k = 0, report half): 24.00%, 24.00%, 24.00%.
Random baseline for most sensitive (logit SQNR) (same k = 16, report half): 25.60%, 30.60%, 24.60%.
