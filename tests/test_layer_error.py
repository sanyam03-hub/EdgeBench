from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    compute_signal_to_quantization_noice_ratio,
    create_activation_matching,
    modify_model_output_intermediate_tensors,
)

from edgebench.data import ImagenetteCalibrationReader, list_images
from edgebench.export import INPUT_NAME, export_onnx
from edgebench.layer_error import (
    NodeScore,
    SqnrAccumulator,
    TensorError,
    WeightError,
    activation_errors,
    array_sqnr_db,
    blame_map,
    choose_k,
    consumer_map,
    quantized_nodes,
    random_exclusions,
    rank_nodes,
    split_halves,
    sqnr_db,
    weight_errors,
)
from edgebench.models import LoadedModel
from edgebench.quantize import preprocess, quantize_static_int8

# --- pure logic -----------------------------------------------------------------------------


def test_sqnr_db_known_value() -> None:
    # ||x|| = 5, ||x - y|| = 0.5 -> 20 * log10(10) = 20 dB
    assert sqnr_db(25.0, 0.25) == pytest.approx(20.0)


def test_sqnr_db_matches_ort_including_its_eps_clamp() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100)
    y = x + 0.01 * rng.standard_normal(100)
    ours = sqnr_db(float(x @ x), float((x - y) @ (x - y)))
    assert ours == pytest.approx(compute_signal_to_quantization_noice_ratio(x, y))
    exact = np.array([3.0, 4.0])  # identical tensors: ORT clamps the noise norm to eps
    assert sqnr_db(25.0, 0.0) == pytest.approx(
        compute_signal_to_quantization_noice_ratio(exact, exact)
    )


def test_array_sqnr_db() -> None:
    ref = np.array([[3.0, 4.0]], dtype=np.float32)
    assert array_sqnr_db(ref, ref + np.float32(0.3)) == pytest.approx(
        compute_signal_to_quantization_noice_ratio(ref.astype(np.float64), ref + 0.3), abs=1e-5
    )
    with pytest.raises(ValueError, match="shape"):
        array_sqnr_db(ref, np.zeros((2, 2)))


def test_accumulator_streaming_equals_one_shot() -> None:
    rng = np.random.default_rng(1)
    chunks = [rng.standard_normal((2, 3)).astype(np.float32) for _ in range(3)]
    noisy = [c + np.float32(0.05) * rng.standard_normal(c.shape).astype(np.float32) for c in chunks]
    acc = SqnrAccumulator()
    for ref, cand in zip(chunks, noisy, strict=True):
        acc.add("t", ref, cand)
    expected = compute_signal_to_quantization_noice_ratio(chunks, noisy)
    assert acc.result() == {"t": pytest.approx(expected)}


def test_accumulator_all_zero_reference_is_undefined() -> None:
    acc = SqnrAccumulator()
    acc.add("dead", np.zeros(4), np.ones(4))  # e.g. a Relu output that is zero everywhere
    assert acc.result() == {"dead": None}


def test_accumulator_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="t"):
        SqnrAccumulator().add("t", np.zeros(3), np.zeros(4))


def _graph() -> onnx.GraphProto:
    nodes = [
        onnx.helper.make_node("Conv", ["x", "w"], ["c"], name="conv"),
        onnx.helper.make_node("Relu", ["c"], ["r"], name="relu"),
        onnx.helper.make_node("Mul", ["r", "r"], ["y"], name="mul"),
        onnx.helper.make_node("Add", ["r", "x"], ["z"], name="add"),
    ]
    vi = [onnx.helper.make_tensor_value_info(n, onnx.TensorProto.FLOAT, None) for n in "xyz"]
    return onnx.helper.make_graph(nodes, "g", vi[:1], vi[1:])


def test_blame_map_folds_fused_relu_into_its_producer() -> None:
    blame = blame_map(_graph())
    assert blame["c"] == ("conv", "Conv", 0)
    assert blame["r"] == ("conv", "Conv", 0)  # Relu is folded into the Conv by the quantizer
    assert blame["y"] == ("mul", "Mul", 2)
    assert "x" not in blame  # graph input: no node to blame


def test_consumer_map_lists_every_consumer_once_in_graph_order() -> None:
    consumers = consumer_map(_graph())
    assert consumers["r"] == ("mul", "add")  # "mul" reads r twice, listed once
    assert consumers["x"] == ("conv", "add")
    assert "y" not in consumers


def test_rank_nodes_blames_activations_on_consumers_and_takes_the_minimum() -> None:
    tensors = [
        TensorError("a", "conv1", "Conv", 0, ("conv2",), local_sqnr_db=30.0, model_sqnr_db=29.0),
        TensorError("b", "conv2", "Conv", 1, ("mul",), local_sqnr_db=5.0, model_sqnr_db=4.0),
        TensorError("c", "mul", "Mul", 2, ("gone",), local_sqnr_db=1.0, model_sqnr_db=1.0),
        TensorError("d", "mul", "Mul", 2, ("conv1",), local_sqnr_db=None, model_sqnr_db=None),
    ]
    weights = [WeightError("w1", "conv1", "Conv", sqnr_db=2.0)]
    eligible = {"conv1": "Conv", "conv2": "Conv", "mul": "Mul"}
    ranked = rank_nodes(tensors, weights, eligible)
    assert ranked == [
        NodeScore("conv1", "Conv", 2.0, "weight"),
        NodeScore("mul", "Mul", 5.0, "activation"),  # its input b is quantized for it
        NodeScore("conv2", "Conv", 30.0, "activation"),
    ]  # "gone" is not eligible; the undefined tensor "d" is ignored


def test_rank_nodes_breaks_ties_by_name() -> None:
    tensors = [
        TensorError("a", None, None, None, ("n2",), local_sqnr_db=7.0, model_sqnr_db=None),
        TensorError("b", None, None, None, ("n1",), local_sqnr_db=7.0, model_sqnr_db=None),
    ]
    ranked = rank_nodes(tensors, [], {"n1": "Add", "n2": "Add"})
    assert [s.node for s in ranked] == ["n1", "n2"]


def test_split_halves_disjoint_deterministic_and_seeded() -> None:
    items = [f"img{i:03d}" for i in range(11)]
    search, report = split_halves(items, seed=0)
    assert len(search) == 5 and len(report) == 6
    assert set(search).isdisjoint(report) and set(search) | set(report) == set(items)
    assert search == sorted(search) and report == sorted(report)
    assert split_halves(items, seed=0) == (search, report)
    assert split_halves(items, seed=1) != (search, report)


def test_random_exclusions_deterministic_subset() -> None:
    eligible = [f"n{i}" for i in range(10)]
    first = random_exclusions(eligible, k=4, seed=0)
    assert len(first) == 4 and set(first) <= set(eligible) and first == sorted(first)
    assert random_exclusions(list(reversed(eligible)), k=4, seed=0) == first
    assert random_exclusions(eligible, k=0, seed=0) == []
    with pytest.raises(ValueError, match="k"):
        random_exclusions(eligible, k=11, seed=0)


def test_choose_k_smallest_within_budget_else_best() -> None:
    curve = {0: 20.0, 1: 60.0, 2: 96.5, 4: 97.0, 8: 97.5}
    assert choose_k(curve, fp32_top1=97.6, max_drop_pts=1.0) == 4
    assert choose_k(curve, fp32_top1=97.6, max_drop_pts=1.5) == 2
    assert choose_k({0: 10.0, 1: 50.0, 2: 50.0}, fp32_top1=97.6, max_drop_pts=1.0) == 1
    with pytest.raises(ValueError, match="empty"):
        choose_k({}, fp32_top1=97.6, max_drop_pts=1.0)


# --- against ORT on tiny_cnn ------------------------------------------------------------------


@pytest.fixture
def models(tiny_cnn: LoadedModel, synthetic_imagenette: Path, tmp_path: Path) -> tuple[Path, Path]:
    """(preprocessed FP32, static QDQ INT8) for tiny_cnn — same node names in both."""
    sample = np.zeros((1, *tiny_cnn.input_shape), dtype=np.float32)
    fp32 = export_onnx(tiny_cnn.module, sample, tmp_path / "onnx_fp32.onnx")
    paths = list_images(synthetic_imagenette / "train")[:8]
    reader = ImagenetteCalibrationReader(INPUT_NAME, paths, tiny_cnn.transforms, 4)
    qdq = quantize_static_int8(fp32, tmp_path / "onnx_int8_static.onnx", reader)
    return preprocess(fp32, tmp_path / "pre.onnx"), qdq


def _batches() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.random((2, 3, 32, 32), dtype=np.float32) for _ in range(3)]


def _ort_reference(pre: Path, qdq: Path, tmp_path: Path) -> dict[str, dict[str, np.ndarray]]:
    """ORT's own (keep-everything) pipeline: matched activations, concatenated, as float64."""
    aug_f, aug_q = tmp_path / "aug_f.onnx", tmp_path / "aug_q.onnx"
    modify_model_output_intermediate_tensors(pre, aug_f)
    modify_model_output_intermediate_tensors(qdq, aug_q)
    feeds = [{INPUT_NAME: batch} for batch in _batches()]
    matched = create_activation_matching(
        collect_activations(str(aug_q), feeds), collect_activations(str(aug_f), feeds)
    )
    return {
        name: {key: np.concatenate(v).astype(np.float64).ravel() for key, v in m.items()}
        for name, m in matched.items()
    }


def _sqnr(x: np.ndarray, y: np.ndarray) -> float | None:
    return None if not x.any() else compute_signal_to_quantization_noice_ratio(x, y)


def test_activation_errors_match_ort_with_relu_aware_reference(
    models: tuple[Path, Path], tmp_path: Path
) -> None:
    pre, qdq = models
    ours = {
        e.tensor: e for e in activation_errors(pre, qdq, _batches(), INPUT_NAME, tmp_path / "w")
    }
    ref = _ort_reference(pre, qdq, tmp_path)
    relu_outputs = {
        o for n in onnx.load(str(pre)).graph.node if n.op_type == "Relu" for o in n.output
    }

    assert ref and set(ours) == set(ref)
    assert relu_outputs & set(ref), "tiny_cnn should have Relu-fused quantized tensors"
    for name, m in ref.items():
        # The QDQ graph drops a Relu after Conv; its uint8 clip does the Relu, so the
        # quantization error must be measured against relu(pre), not the raw Conv output.
        local_ref = np.maximum(m["pre_qdq"], 0) if name in relu_outputs else m["pre_qdq"]
        expected_local, expected_model = (
            _sqnr(local_ref, m["post_qdq"]),
            _sqnr(m["float"], m["post_qdq"]),
        )
        assert ours[name].local_sqnr_db == pytest.approx(expected_local, abs=1e-9)
        assert ours[name].model_sqnr_db == pytest.approx(expected_model, abs=1e-9)
    graph = onnx.load(str(pre)).graph
    names = {n.name for n in graph.node}
    assert {c for e in ours.values() for c in e.consumers} <= names
    assert ours[graph.input[0].name].node is None  # the model input has no producer


def test_weight_errors_cover_conv_weights(models: tuple[Path, Path]) -> None:
    pre, qdq = models
    errors = weight_errors(pre, qdq)
    conv_nodes = {n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"}
    assert conv_nodes <= {e.node for e in errors}
    assert all(math.isfinite(e.sqnr_db) and e.sqnr_db > 0 for e in errors)


def test_quantized_nodes_maps_compute_nodes_to_op_types(models: tuple[Path, Path]) -> None:
    pre, qdq = models
    nodes = quantized_nodes(qdq)
    assert nodes and not {"QuantizeLinear", "DequantizeLinear"} & set(nodes.values())
    convs = {n.name for n in onnx.load(str(pre)).graph.node if n.op_type == "Conv"}
    assert {n: nodes[n] for n in convs} == dict.fromkeys(convs, "Conv")
