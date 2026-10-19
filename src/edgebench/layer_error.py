"""Per-layer INT8 error analysis (stretch S1): which layers lose the signal when quantized.

Compares an FP32 model with its static QDQ INT8 model tensor by tensor. ORT's
``qdq_loss_debug`` does the graph augmentation and the FP32 <-> QDQ name matching, but its
``collect_activations`` keeps every activation of every batch in memory (~38 MB per image for
MobileNetV3-Small). Here the error is accumulated batch by batch instead, so memory stays at
one batch; apart from the two corrections below the numbers equal ORT's (tested).

Error measure: SQNR = 20*log10(||x|| / ||x - x_hat||) in dB (higher is better; ~6 dB per bit).

- ``local_sqnr_db``: a tensor before vs after its own Quantize/Dequantize inside the INT8 model —
  the rounding/clipping that quantizing *this* tensor adds.
- ``model_sqnr_db``: the FP32 model's tensor vs the INT8 model's — error accumulated through all
  earlier layers. Shows where the two models diverge (it grows with depth).

Two corrections to ORT's ``compute_activation_error``:
- The QDQ quantizer removes a Relu that follows a Conv and lets the uint8 clip (zero point 0)
  do it, so ORT compares the *raw Conv output* with the clipped tensor and reports the Relu
  itself as error (MobileNetV3-Small, 4 images: median 3.8 dB, vs 40.7 dB against relu(x)). Here the
  reference for a Relu output is relu(pre-QDQ tensor).
- SQNR of an all-zero reference is undefined (ORT's eps clamp gives 0 dB or ~-300 dB); it is
  reported as None and never ranked.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization.qdq_loss_debug import (
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors,
)

T = TypeVar("T")

EPS = float(np.finfo(float).eps)  # same clamp as ORT's compute_signal_to_quantization_noice_ratio
SAVED_SUFFIX = "_ReshapedSavedOutput"  # suffix ORT's augmentation gives the saved tensors
# The QDQ quantizer folds these into the op that produces their input (e.g. Conv+Relu).
FUSED_ACTIVATIONS = frozenset({"Relu", "Clip"})
_QDQ_OPS = frozenset({"QuantizeLinear", "DequantizeLinear"})


@dataclass(frozen=True)
class TensorError:
    """SQNR of one activation tensor of the FP32 graph."""

    tensor: str
    node: str | None  # producing node (fused Relu folded into its producer); None: graph input
    op_type: str | None
    depth: int | None  # index of ``node`` in the (topologically sorted) FP32 graph
    consumers: tuple[str, ...]  # nodes that read it: where its quantization is applied
    local_sqnr_db: float | None  # None: all-zero reference (undefined)
    model_sqnr_db: float | None  # None: undefined, or no FP32 tensor of that name


@dataclass(frozen=True)
class WeightError:
    """SQNR of one stored weight (FP32 vs dequantized INT8) and the node that uses it."""

    weight: str
    node: str
    op_type: str
    sqnr_db: float


@dataclass(frozen=True)
class NodeScore:
    """A node's score for ranking: lower SQNR = keep it in FP32 first."""

    node: str
    op_type: str
    sqnr_db: float
    source: str  # "weight" | "activation" (an input it reads) | "logits" (sensitivity)


def sqnr_db(signal_sq: float, noise_sq: float) -> float:
    """SQNR in dB from ||x||^2 and ||x - y||^2, with ORT's eps clamp on both norms."""
    return 20 * math.log10(max(math.sqrt(signal_sq), EPS) / max(math.sqrt(noise_sq), EPS))


def array_sqnr_db(reference: np.ndarray, candidate: np.ndarray) -> float:
    """SQNR in dB of ``candidate`` against ``reference`` (same shape), computed in float64."""
    if np.shape(reference) != np.shape(candidate):
        raise ValueError(f"shape mismatch: {np.shape(reference)} vs {np.shape(candidate)}")
    ref = np.asarray(reference, dtype=np.float64).ravel()
    diff = ref - np.asarray(candidate, dtype=np.float64).ravel()
    return sqnr_db(float(ref @ ref), float(diff @ diff))


class SqnrAccumulator:
    """Sums squared norms per tensor name, so SQNR over many batches needs one batch of memory."""

    def __init__(self) -> None:
        self._signal: dict[str, float] = {}
        self._noise: dict[str, float] = {}

    def add(self, name: str, reference: np.ndarray, candidate: np.ndarray) -> None:
        ref = np.asarray(reference, dtype=np.float64).ravel()
        cand = np.asarray(candidate, dtype=np.float64).ravel()
        if ref.size != cand.size:
            raise ValueError(f"{name}: reference has {ref.size} values, candidate {cand.size}")
        diff = ref - cand
        self._signal[name] = self._signal.get(name, 0.0) + float(ref @ ref)
        self._noise[name] = self._noise.get(name, 0.0) + float(diff @ diff)

    def result(self) -> dict[str, float | None]:
        """SQNR per name; None where the reference was zero everywhere (undefined)."""
        return {
            name: sqnr_db(signal, self._noise[name]) if signal > 0 else None
            for name, signal in self._signal.items()
        }


def blame_map(graph: onnx.GraphProto) -> dict[str, tuple[str, str, int]]:
    """Tensor -> (node, op_type, depth) of the node responsible for producing it.

    Outputs of a fused activation (Relu/Clip) are blamed on the node producing its input,
    because after quantization that node writes the clipped tensor directly.
    """
    producer: dict[str, tuple[str, str, int]] = {}
    for depth, node in enumerate(graph.node):
        for output in node.output:
            producer[output] = (node.name, node.op_type, depth)
    blame = dict(producer)
    for node in graph.node:
        if node.op_type in FUSED_ACTIVATIONS and node.input and node.input[0] in producer:
            for output in node.output:
                blame[output] = producer[node.input[0]]
    return blame


def consumer_map(graph: onnx.GraphProto) -> dict[str, tuple[str, ...]]:
    """Tensor -> names of the nodes that read it, in graph order, each once."""
    consumers: dict[str, list[str]] = {}
    for node in graph.node:
        for name in dict.fromkeys(node.input):
            if name:
                consumers.setdefault(name, []).append(node.name)
    return {name: tuple(nodes) for name, nodes in consumers.items()}


def activation_errors(
    float_model: Path,
    qdq_model: Path,
    batches: Iterable[np.ndarray],
    input_name: str,
    workdir: Path,
) -> list[TensorError]:
    """Local and cross-model SQNR for every quantized activation, accumulated over ``batches``.

    Args:
        float_model: The FP32 model *as fed to the quantizer* (``quantize.preprocess`` output),
            so tensor and node names match the QDQ model.
        qdq_model: Static QDQ INT8 model.
        batches: float32 (N, 3, H, W) arrays.
        input_name: Model input name.
        workdir: Where the two augmented models are written.
    """
    graph = onnx.load(str(float_model)).graph
    relu_outputs = {out for node in graph.node if node.op_type == "Relu" for out in node.output}
    workdir.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path, tag in ((float_model, "float"), (qdq_model, "qdq")):
        augmented = workdir / f"{tag}_augmented.onnx"
        modify_model_output_intermediate_tensors(path, augmented)
        options = ort.SessionOptions()
        # As in ORT's collect_activations: no fusion, so every saved tensor still exists.
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sessions.append(
            ort.InferenceSession(str(augmented), options, providers=["CPUExecutionProvider"])
        )

    local, model = SqnrAccumulator(), SqnrAccumulator()
    for batch in batches:
        float_acts, qdq_acts = (_saved_tensors(s, {input_name: batch}) for s in sessions)
        for name, match in create_activation_matching(qdq_acts, float_acts).items():
            pre, post = match["pre_qdq"][0], match["post_qdq"][0]
            local.add(name, np.maximum(pre, 0) if name in relu_outputs else pre, post)
            if "float" in match:
                model.add(name, match["float"][0], post)

    blame, consumers = blame_map(graph), consumer_map(graph)
    model_sqnr = model.result()
    errors = []
    for name, value in sorted(local.result().items()):
        node, op_type, depth = blame.get(name, (None, None, None))
        errors.append(
            TensorError(
                name, node, op_type, depth, consumers.get(name, ()), value, model_sqnr.get(name)
            )
        )
    return errors


def _saved_tensors(
    session: ort.InferenceSession, feed: dict[str, np.ndarray]
) -> dict[str, list[np.ndarray]]:
    names = [output.name for output in session.get_outputs()]
    values = session.run(None, feed)
    return {
        name[: -len(SAVED_SUFFIX)]: [value]
        for name, value in zip(names, values, strict=True)
        if name.endswith(SAVED_SUFFIX)
    }


def weight_errors(float_model: Path, qdq_model: Path) -> list[WeightError]:
    """SQNR of every quantized weight, attributed to the first FP32 node that consumes it."""
    sqnr = compute_weight_error(create_weight_matching(str(float_model), str(qdq_model)))
    consumer: dict[str, tuple[str, str]] = {}
    for node in onnx.load(str(float_model)).graph.node:
        for name in node.input:
            consumer.setdefault(name, (node.name, node.op_type))
    return [
        WeightError(weight, *consumer[weight], value)
        for weight, value in sorted(sqnr.items())
        if weight in consumer
    ]


def quantized_nodes(qdq_model: Path) -> dict[str, str]:
    """Compute nodes that read at least one dequantized (INT8) tensor -> their op type.

    These are the nodes ``nodes_to_exclude`` can usefully keep in FP32.
    """
    graph = onnx.load(str(qdq_model)).graph
    dq_outputs = {n.output[0] for n in graph.node if n.op_type == "DequantizeLinear"}
    return {
        node.name: node.op_type
        for node in graph.node
        if node.op_type not in _QDQ_OPS and any(name in dq_outputs for name in node.input)
    }


def rank_nodes(
    tensor_errors: Iterable[TensorError],
    weight_errors: Iterable[WeightError],
    eligible: Mapping[str, str],
) -> list[NodeScore]:
    """Eligible nodes by local error, worst (lowest SQNR) first; ties broken by node name.

    A node's score is the minimum over its weights and the activations it *reads*: keeping a
    node in FP32 removes the quantization of its weights and of its inputs (for that node), not
    of its output, which its consumers still quantize.
    """
    worst: dict[str, NodeScore] = {}

    def consider(node: str, value: float, source: str) -> None:
        if node in eligible and (node not in worst or value < worst[node].sqnr_db):
            worst[node] = NodeScore(node, eligible[node], value, source)

    for weight in weight_errors:
        consider(weight.node, weight.sqnr_db, "weight")
    for tensor in tensor_errors:
        if tensor.local_sqnr_db is not None:
            for consumer in tensor.consumers:
                consider(consumer, tensor.local_sqnr_db, "activation")
    return sorted(worst.values(), key=lambda score: (score.sqnr_db, score.node))


def split_halves(items: Sequence[T], seed: int) -> tuple[list[T], list[T]]:
    """Seeded random split into (search, report); each half sorted. Search gets the smaller half."""
    order = np.random.default_rng(seed).permutation(len(items))
    half = len(items) // 2
    search = sorted(items[i] for i in order[:half])  # type: ignore[type-var]
    report = sorted(items[i] for i in order[half:])  # type: ignore[type-var]
    return search, report


def random_exclusions(eligible: Collection[str], k: int, seed: int) -> list[str]:
    """``k`` distinct nodes drawn uniformly from ``eligible`` (seeded by ``seed`` and ``k``)."""
    pool = sorted(set(eligible))
    if not 0 <= k <= len(pool):
        raise ValueError(f"k={k} must be between 0 and the {len(pool)} eligible nodes")
    chosen = np.random.default_rng([seed, k]).choice(len(pool), size=k, replace=False)
    return sorted(pool[i] for i in chosen)


def choose_k(curve: Mapping[int, float], fp32_top1: float, max_drop_pts: float) -> int:
    """Smallest k with top-1 within ``max_drop_pts`` of FP32; else the best k (smallest on ties)."""
    if not curve:
        raise ValueError("choose_k: the accuracy curve is empty")
    ks = sorted(curve)
    within = [k for k in ks if curve[k] >= fp32_top1 - max_drop_pts - 1e-9]
    return within[0] if within else max(ks, key=lambda k: curve[k])
