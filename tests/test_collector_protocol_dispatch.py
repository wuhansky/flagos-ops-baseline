"""Keep custom timing distinct from preparation and reset protocols."""
import copy
from types import SimpleNamespace
import pytest
from test_prepared_sample_timing import Harness


def test_custom_measurement_preserves_native_arguments_and_record_units():
    h = Harness()
    ns = h.load()
    calls = []
    def measure(op, args, kwargs):
        assert op == h.native and (args, kwargs) == h.original
        calls.append(1)
        op(*args, **kwargs)
        return 3.5
    recipe = SimpleNamespace(
        OP_NAME="custom", __name__="custom", native=lambda: h.native,
        DTYPES=["torch.float32"], grid=lambda: [{}], key_shape=lambda b: "case",
        build_inputs=lambda *args: copy.deepcopy(h.original),
        config=lambda *args: {"original": True}, measure_latency=measure,
    )
    record = ns["_collect_one_op_ops"](recipe, False, None)["shapes"]["case"]["torch.float32"]
    assert calls == [1] and h.native_calls == 2  # preflight, then custom timing
    assert record["T_us"] == 3500 and record["latency_ms"] == 3.5
    assert record["config"] == {"original": True} and not h.default_calls


@pytest.mark.parametrize("other", ["prepare_sample", "make_input_reset"])
def test_ambiguous_protocols_rejected_before_native_resolution(other):
    h = Harness()
    def forbidden():
        pytest.fail("must reject ambiguous timing before resolving native")
    recipe = SimpleNamespace(OP_NAME="custom", native=forbidden, measure_latency=lambda *a: 1)
    setattr(recipe, other, lambda *a: None)
    with pytest.raises(ValueError, match="cannot be combined"):
        h.load()["_collect_one_op_ops"](recipe, False, None)
    assert h.log == []


def test_invalid_custom_timer_rejected_before_native_resolution():
    h = Harness()
    recipe = SimpleNamespace(OP_NAME="custom", measure_latency=42)
    with pytest.raises(TypeError, match="measure_latency must be callable"):
        h.load()["_collect_one_op_ops"](recipe, False, None)
    assert h.log == []


def test_custom_timer_cannot_be_mislabeled_as_a_return_mode_override():
    h = Harness()
    recipe = SimpleNamespace(OP_NAME="custom", measure_latency=lambda *a: 1,
                             BENCH_RETURN_MODE="median")
    with pytest.raises(ValueError, match="return-mode overrides"):
        h.load()["_collect_one_op_ops"](recipe, False, None)
    assert h.log == []
