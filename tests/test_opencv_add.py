"""OpenCV add contract and opt-in real CUDA/stream checks."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('opencv_add_profile', ROOT / 'ops/opencv/add.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_keys_and_missing_native(monkeypatch):
    keys = [str(profile.key_shape(b)) for b in profile.grid()]
    assert len(keys) == len(set(keys)) == 4
    monkeypatch.delenv('OPENCV_BASELINE_LIBRARY', raising=False)
    with pytest.warns(UserWarning, match='OPENCV_BASELINE_LIBRARY'):
        assert profile.native() is None


@pytest.mark.skipif(not torch.cuda.is_available() or not os.getenv('OPENCV_BASELINE_LIBRARY'),
                    reason='requires real CUDA OpenCV bridge')
@pytest.mark.parametrize('binding', profile.grid())
@pytest.mark.parametrize('dtype', profile.DTYPES)
def test_real_native(binding, dtype):
    op = profile.native()
    assert op is not None
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        (a, b, out), kwargs = profile.build_inputs(binding, dtype, 'cuda')
        # Include overflowing uint8 addition, which must saturate rather than wrap.
        if dtype == torch.uint8:
            a.view(-1)[:4] = torch.tensor([0, 1, 200, 255], device='cuda', dtype=dtype)
            b.view(-1)[:4] = torch.tensor([0, 255, 100, 1], device='cuda', dtype=dtype)
        out.fill_(37)
        a_before, b_before = a.clone(), b.clone()
        result = op(a, b, out)
        expected = ((a.to(torch.int16) + b.to(torch.int16)).clamp(0, 255).to(dtype)
                    if dtype == torch.uint8 else a + b)
        assert result is out
    stream.synchronize()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert torch.equal(a, a_before) and torch.equal(b, b_before)
    # Repeated invocations must overwrite output, not accumulate into it.
    with torch.cuda.stream(stream):
        a.zero_()
        op(a, b, out)
    stream.synchronize()
    torch.testing.assert_close(out, b, rtol=0, atol=0)
    with pytest.raises(ValueError, match='separate'):
        op(a, b, a)
    assert profile.config(binding, dtype)['native_api'] == 'cv::cuda::add'
