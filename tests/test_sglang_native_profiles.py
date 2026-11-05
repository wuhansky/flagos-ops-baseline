"""Exact document layouts are retained without loading SGLang or a GPU."""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/sglang_native"))
import _sglang_profiles  # noqa: E402


def test_documented_case_inventory_and_no_key_collisions():
    if not os.environ.get("FLAGGEMS_SGLANG_SOURCE"):
        pytest.skip("set FLAGGEMS_SGLANG_SOURCE for paired source validation")
    expected = {
        "gemma_rmsnorm": 12,
        "gemma_fused_add_rmsnorm": 4,
        "silu_and_mul_out": 4,
        "select_experts": 2,
        "_fused_moe_kernel_sequence": 2,
        "chunk_gated_delta_rule_fwd": 2,
        "fused_recurrent_gated_delta_rule_packed_decode": 2,
    }
    for op, count in expected.items():
        rows = _sglang_profiles.grid(op)
        assert len(rows) == count
        assert len({_sglang_profiles.key_shape(op, row) for row in rows}) == count
    for hidden, stride, offset in [(5120, 3584, 3072), (2048, 2560, 2048)]:
        row = next(
            r
            for r in _sglang_profiles.grid("gemma_rmsnorm")
            if r["hidden"] == hidden and r["role"] == "k" and r["tokens"] == 20
        )
        _, values = _sglang_profiles.inputs(
            "gemma_rmsnorm", row, torch.bfloat16, "meta"
        )
        assert values["input"].stride() == (stride, 1)
        assert values["input"].storage_offset() == offset


def test_native_never_substitutes_a_torch_reference(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_SGLANG_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="FLAGGEMS_SGLANG_SOURCE"):
        _sglang_profiles.native("gemma_rmsnorm")


def test_profiles_are_discoverable_by_the_real_collector():
    import importlib.util

    pytest.importorskip("triton", reason="collector requires the Linux runtime")
    # Load the real host collector rather than reproducing its module contract.
    spec = importlib.util.spec_from_file_location(
        "native_collector", ROOT / "tools/collect_baseline_nvidia.py"
    )
    collector = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    spec.loader.exec_module(collector)
    found = dict(collector._discover_ops(ROOT / "ops/sglang_native"))
    assert len(found) == 7
    assert all(callable(module.native) for module in found.values())
    assert all(module.BENCH_RETURN_MODE == "median" for module in found.values())


def test_gdn_reported_head_counts_cannot_contradict_tensor_geometry():
    if not os.environ.get("FLAGGEMS_SGLANG_SOURCE"):
        pytest.skip("set FLAGGEMS_SGLANG_SOURCE for paired source validation")
    operator = "chunk_gated_delta_rule_fwd"
    for row in _sglang_profiles.grid(operator):
        cfg = _sglang_profiles.config(operator, row, torch.bfloat16)
        params = cfg["parameters"]
        q_heads = cfg["inputs"]["q"]["shape"][-2]
        v_heads = cfg["inputs"]["v"]["shape"][-2]
        assert params.get("q_heads", q_heads) == q_heads
        assert params.get("value_heads", v_heads) == v_heads


def test_published_operator_names_match_collector_recipes():
    import importlib.util
    import json

    artifact = json.loads(
        (ROOT / "op_sglang_native_h800_perf_baseline.json").read_text()
    )
    names = set()
    for path in sorted((ROOT / "ops/sglang_native").glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location("_recipe_" + path.stem, path)
        recipe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(recipe)
        assert recipe.OP_NAME == "sglang_fl_" + recipe._NATIVE_OPERATOR.lstrip("_")
        assert recipe.OP_NAME in artifact
        assert recipe._NATIVE_OPERATOR not in artifact
        names.add(recipe.OP_NAME)
    assert len(names) == 7
