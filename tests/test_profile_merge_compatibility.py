"""Explicit profiles preserve separate catalogs and scalar/native overloads."""
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

OPS = Path(__file__).resolve().parents[1] / "ops/pytorch"
sys.path.insert(0, str(OPS))


def load(name, selector, value):
    env = dict(os.environ)
    env.pop(selector, None)
    if value is not None:
        env[selector] = value
    spec = importlib.util.spec_from_file_location("profile_contract_" + name, OPS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    with patch.dict(os.environ, env, clear=True):
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value,count", [(None, 14), ("0", 9), ("1", 5)])
def test_median_profiles_keep_all_cases_and_no_dim_overload(value, count):
    module = load("median", "OPS_BASELINE_MEDIAN_SUIYUAN_CORE", value)
    grid = module.grid()
    assert len(grid) == count and len({str(module.key_shape(b)) for b in grid}) == count
    assert len(module.DTYPES) == 5 and not module.IS_INPLACE
    for binding in grid:
        for dtype in module.DTYPES:
            args, kwargs = module.build_inputs(binding, dtype, "meta")
            assert len(args) == 1 and kwargs == {} and args[0].dtype == dtype
            assert module.config(binding, dtype)
    binding = next(b for b in grid if torch.tensor(b["shape"]).prod().item() <= 4096)
    with torch.random.fork_rng(devices=[]):
        args, kwargs = module.build_inputs(binding, torch.float32, "cpu")
        result = module.native()(*args, **kwargs)
        assert result.shape == ()
        torch.testing.assert_close(result, torch.median(args[0]))


def test_frozen_median_inventory_selects_its_historical_profile():
    info = json.loads((OPS / "_kernelgen/manifest.json").read_text())["operators"]["median"]
    assert info["producer_selector"] == {"OPS_BASELINE_MEDIAN_SUIYUAN_CORE": "0"}
    module = load("median", "OPS_BASELINE_MEDIAN_SUIYUAN_CORE", "0")
    assert len(module.grid()) == info["cases"]


@pytest.mark.parametrize("value,count", [(None, 9), ("0", 9), ("1", 15)])
def test_sdpa_profiles_keep_distinct_case_keys(value, count):
    module = load("_scaled_dot_product_flash_attention", "OPS_BASELINE_SDPA_INSPECT216", value)
    grid = module.grid()
    assert len(grid) == count and len({str(module.key_shape(b)) for b in grid}) == count
    assert module.native() is torch.ops.aten._scaled_dot_product_flash_attention.default
    for binding in grid:
        for dtype in module.DTYPES:
            args, kwargs = module.build_inputs(binding, dtype, "meta")
            assert args[0].dtype == dtype and module.config(binding, dtype)


@pytest.mark.parametrize("name,selector", [
    ("median", "OPS_BASELINE_MEDIAN_SUIYUAN_CORE"),
    ("_scaled_dot_product_flash_attention", "OPS_BASELINE_SDPA_INSPECT216"),
])
def test_unknown_profile_rejected(name, selector):
    with pytest.raises(ValueError):
        load(name, selector, "invalid")
