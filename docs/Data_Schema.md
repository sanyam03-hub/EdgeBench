# Data Schema — Configs, Results, Baselines and Interfaces

> Replaces the PDF's "Database" and "Backend Schema" documents. EdgeBench has **no database, no server, no auth and no user roles** (so no tables, RLS, or API endpoints). Its persistent data is YAML configs and JSON files, and its "backend" contract is the Python `Backend` interface. Those are defined here. Field names here are the source of truth — code must match them.
>
> All example values below are **illustrative placeholders**, not measured results.

## 1. Why no database
Each run produces one immutable JSON file. Files are diffable in git, readable without tools, and enough for comparing a handful of runs. A database would add setup cost with no benefit at this scale. Revisit only if there are thousands of runs to query.

## 2. Model config — `configs/<model>.yaml`
```yaml
schema_version: 1
model: mobilenet_v3_small          # key in models.py registry
weights: DEFAULT                   # torchvision weights enum member
dataset: imagenette2-160           # or "synthetic" (smoke config only)
variants:                          # fixed vocabulary, see §6
  - torch_fp32
  - onnx_fp32
  - onnx_int8_dynamic
  - onnx_int8_static
export:
  opset: 17
quantization:
  static:
    per_channel: true
    calibrate_method: minmax       # minmax | entropy | percentile
    activation_type: quint8
    weight_type: qint8
  calibration:                     # drawn ONLY from train/
    n_images: 256
    batch_size: 32
    seed: 0
eval:                              # drawn ONLY from val/
  n_images: 1000
  batch_size: 32
  seed: 0
parity:
  n_images: 32                     # first batch of the eval subset
  fp32:                            # applies to onnx_fp32, tflite_fp32
    max_abs_err: 1.0e-3            # initial value; calibrate in Phase 4 and record why
    min_cosine: 0.9999
    min_top1_agreement: 0.99
  int8:                            # applies to *_int8_* variants
    min_top1_agreement: 0.90       # no max_abs_err check for INT8
```

## 3. Target profile — `configs/targets/<target>.yaml`
```yaml
schema_version: 1
name: cpu_4t
intra_op_threads: 4                # ORT intra-op and torch.set_num_threads
inter_op_threads: 1
batch_sizes: [1, 8]
warmup: 20
runs: 200
noisy_cv_threshold: 0.10           # CV above this marks a measurement as noisy
```
`ci.yaml` uses small values (e.g. `batch_sizes: [1]`, `warmup: 2`, `runs: 10`) so the smoke run is fast.

## 4. Results file — `results/<run_id>.json`
- Encoding UTF-8, `indent=2`, `sort_keys=True`.
- Timestamps: ISO 8601 UTC with `Z` suffix, e.g. `2026-01-01T12:00:00Z`.
- `run_id`: `YYYYMMDDTHHMMSSZ_<first 8 hex chars of SHA-256 of the canonical config JSON>`.
- Paths are **relative to the project root**.

```json
{
  "schema_version": 1,
  "run_id": "20260101T120000Z_ab12cd34",
  "created_at": "2026-01-01T12:00:00Z",
  "edgebench_version": "0.1.0",
  "config_hash": "ab12cd34...",
  "config": { "model": { "...": "full model config snapshot" }, "target": { "...": "full target snapshot" } },
  "environment": {
    "os": "Linux",
    "kernel": "5.15.x-microsoft-standard-WSL2",
    "cpu_model": "<from /proc/cpuinfo>",
    "logical_cores": 8,
    "python": "3.11.x",
    "packages": { "torch": "x.y.z", "torchvision": "x.y.z", "onnx": "x.y.z", "onnxruntime": "x.y.z", "numpy": "x.y.z" }
  },
  "variants": [
    {
      "model": "mobilenet_v3_small",
      "variant": "onnx_int8_static",
      "runtime": "onnxruntime",
      "precision": "int8_static",
      "status": "ok",
      "error": null,
      "artifact": "artifacts/mobilenet_v3_small/onnx_int8_static.onnx",
      "size_mb": 0.0,
      "load_ms": 0.0,
      "parity": {
        "n_samples": 32,
        "max_abs_err": 0.0,
        "mean_abs_err": 0.0,
        "cosine_similarity": 0.0,
        "top1_agreement": 0.0,
        "within_tolerance": true,
        "tolerance": { "min_top1_agreement": 0.9 }
      },
      "accuracy": {
        "top1": 0.0,
        "n_images": 1000,
        "delta_vs_onnx_fp32_pts": 0.0
      },
      "latency": {
        "1": { "n": 200, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "std_ms": 0.0, "iqr_ms": 0.0, "cv": 0.0, "noisy": false },
        "8": { "n": 200, "p50_ms": 0.0, "p90_ms": 0.0, "p99_ms": 0.0, "mean_ms": 0.0, "std_ms": 0.0, "iqr_ms": 0.0, "cv": 0.0, "noisy": false }
      }
    }
  ]
}
```

Field rules
| Field | Type | Rule |
|---|---|---|
| `status` | `"ok" \| "error" \| "skipped"` | `error`/`skipped` → `error` holds `"<ExceptionType>: <message>"` or the skip reason; `parity`, `accuracy`, `latency` are `null` |
| `torch_fp32` row | — | `parity` is `null` (it is the reference) |
| `accuracy.top1` | float, 0–100 | Percent, see §5 |
| `delta_vs_onnx_fp32_pts` | float | `top1 − top1(onnx_fp32)`; `null` for `torch_fp32` and when `onnx_fp32` failed |
| `latency` keys | string batch sizes | One entry per `batch_sizes` value |
| `cv` | float | `std / mean` (fraction, not percent) |
| `size_mb` | float | File size / 1e6 |
| `load_ms` | float | Wall time to create the session/interpreter/model |

## 5. Accuracy definition (fixed)
Pretrained models output 1000 ImageNet logits. Imagenette has 10 classes that map to ImageNet indices `[0, 217, 482, 491, 497, 566, 569, 571, 574, 701]`.
**Top-1 = argmax over only those 10 logits**, compared with the image's label. This is "10-way accuracy among Imagenette classes"; state this in the report and README. It is **not** ImageNet top-1 and must not be compared to published ImageNet numbers.
For the `tiny_cnn` smoke config, the model outputs 10 logits directly and data is synthetic, so its accuracy is meaningless — the smoke run tests pipeline mechanics only.

## 6. Variant vocabulary
| Variant | Runtime | Precision | Artifact | Backend class |
|---|---|---|---|---|
| `torch_fp32` | torch | fp32 | — (in-memory model) | `TorchBackend` |
| `onnx_fp32` | onnxruntime | fp32 | `onnx_fp32.onnx` | `OrtBackend` |
| `onnx_int8_dynamic` | onnxruntime | int8_dynamic | `onnx_int8_dynamic.onnx` | `OrtBackend` |
| `onnx_int8_static` | onnxruntime | int8_static | `onnx_int8_static.onnx` | `OrtBackend` |
| `tflite_fp32` | litert | fp32 | `tflite_fp32.tflite` | `TFLiteBackend` (optional) |
| `tflite_int8_dynamic` | litert | int8_dynamic | `tflite_int8_dynamic.tflite` | `TFLiteBackend` (stretch) |

## 7. Baseline file — `baselines/<target>.json`
```json
{
  "schema_version": 1,
  "target": "ci",
  "created_from_run_id": "20260101T120000Z_ab12cd34",
  "created_at": "2026-01-01T12:00:00Z",
  "thresholds": { "max_accuracy_drop_pts": 0.5 },
  "entries": [
    { "model": "tiny_cnn", "variant": "onnx_fp32", "top1": 0.0, "require_parity": true },
    { "model": "tiny_cnn", "variant": "onnx_int8_static", "top1": 0.0, "require_parity": true }
  ]
}
```
Gate checks per entry: variant present → `status == "ok"` → `top1 >= baseline.top1 − max_accuracy_drop_pts` → if `require_parity`, `parity.within_tolerance` is true. Latency: printed, never checked.
Create the CI baseline from a smoke run **on the CI runner** (download the CI artifact), not from the local machine, so platform differences don't cause false failures.

## 8. Python interfaces (in-code contracts)

```python
# src/edgebench/backends/base.py
class Backend(Protocol):
    variant: str          # e.g. "onnx_int8_static"
    runtime: str          # "torch" | "onnxruntime" | "litert"
    precision: str        # "fp32" | "int8_dynamic" | "int8_static"

    def predict(self, x: np.ndarray) -> np.ndarray:
        """x: float32, shape (N, 3, H, W). Returns float32 logits (N, C)."""

# src/edgebench/backends/registry.py
def create_backend(variant: str, *, model_name: str, artifact: Path | None,
                   target: TargetConfig) -> tuple[Backend, float]:
    """Return the backend and its load time in ms. Raises BackendError."""
```

```python
@dataclass(frozen=True)
class ParityReport:
    n_samples: int
    max_abs_err: float
    mean_abs_err: float
    cosine_similarity: float     # mean of per-row cosine similarities
    top1_agreement: float        # fraction 0–1
    within_tolerance: bool
    tolerance: dict[str, float]

@dataclass(frozen=True)
class LatencyStats:
    n: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    std_ms: float
    iqr_ms: float
    cv: float
    noisy: bool

@dataclass(frozen=True)
class GateCheck:
    model: str
    variant: str
    check: str                   # "present" | "status" | "accuracy" | "parity" | "latency_info"
    passed: bool
    detail: str
```

## 9. File storage layout
| Path | Contents | In git? |
|---|---|---|
| `data/imagenette2-160/{train,val}/<wnid>/*.JPEG` | Dataset | No |
| `artifacts/<model>/<variant>.{onnx,tflite}` | Exported models | No |
| `results/<run_id>.json` | Run results | No |
| `results/<run_id>/report.md`, `pareto.png` | Reports | No |
| `results/sample/` | One curated run used in the README | Yes |
| `baselines/<target>.json` | Gate baselines | Yes |

## 10. Schema versioning
- Every config, results and baseline file has `schema_version`. Current version: **1**.
- Loading a file with a different version raises `SchemaError` with the expected and found versions.
- Any field change bumps the version and updates this document in the same commit.
