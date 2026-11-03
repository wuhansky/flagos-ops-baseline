# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""OpenCV CUDA add, same-type matrices, one channel, no mask.

Native API: cv::cuda::add in opencv_contrib/modules/cudaarithm.
Workloads are synthetic image sizes, not copied from an existing benchmark.
Device-resident inputs/output and the Torch current stream are passed through a
small C ABI bridge. Allocation and host/device transfers are outside timing.
"""
import ctypes
import hashlib
import os
from pathlib import Path
import warnings

import torch

OP_NAME = 'add'
DTYPES = [torch.float32, torch.uint8]
IS_INPLACE = False


def _library():
    path = os.environ.get('OPENCV_BASELINE_LIBRARY')
    if not path:
        raise RuntimeError('Set OPENCV_BASELINE_LIBRARY to the built OpenCV CUDA bridge')
    lib = ctypes.CDLL(path)
    lib.opencv_add.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    lib.opencv_add.restype = ctypes.c_int
    for symbol in ('opencv_version', 'opencv_build_info', 'opencv_error'):
        getattr(lib, symbol).restype = ctypes.c_char_p
    return lib


def native():
    try:
        lib = _library()
    except (OSError, RuntimeError, AttributeError) as exc:
        warnings.warn(f'OpenCV CUDA unavailable: {exc}')
        return None

    def add(a, b, out):
        if not (a.is_cuda and b.device == a.device and out.device == a.device):
            raise ValueError('add requires tensors on the same CUDA device')
        if not (a.ndim == 2 and a.shape == b.shape == out.shape):
            raise ValueError('add requires equal 2D shapes')
        if not (a.dtype in DTYPES and a.dtype == b.dtype == out.dtype):
            raise ValueError('add supports same-type float32/uint8 tensors')
        if not all(x.is_contiguous() for x in (a, b, out)):
            raise ValueError('add requires contiguous tensors')
        if any(out.data_ptr() < x.data_ptr() + x.numel() * x.element_size()
               and x.data_ptr() < out.data_ptr() + out.numel() * out.element_size()
               for x in (a, b)):
            raise ValueError('output must be separate from inputs')
        code = lib.opencv_add(a.data_ptr(), b.data_ptr(), out.data_ptr(),
                              a.shape[0], a.shape[1],
                              5 if a.dtype == torch.float32 else 0,
                              a.device.index, torch.cuda.current_stream(a.device).cuda_stream)
        if code:
            raise RuntimeError(lib.opencv_error().decode())
        return out
    return add


def grid():
    return [{'shape': [h, w]} for h, w in [(64, 64), (480, 640), (1080, 1920), (2160, 3840)]]


def build_inputs(binding, dtype, device):
    shape = binding['shape']
    if dtype == torch.uint8:
        a = torch.randint(0, 256, shape, dtype=dtype, device=device)
        b = torch.randint(0, 256, shape, dtype=dtype, device=device)
    else:
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn_like(a)
    return (a, b, torch.empty_like(a)), {}


def key_shape(binding):
    return list(binding['shape'])


def config(binding, dtype):
    lib = _library()
    path = Path(os.environ['OPENCV_BASELINE_LIBRARY']).resolve()
    return {'native_api': 'cv::cuda::add', 'channels': 1, 'mask': None,
            'output_dtype': str(dtype), 'output_preallocated': True,
            'stream': 'torch.cuda.current_stream',
            'workload_source': 'synthetic image dimensions',
            'opencv_version': lib.opencv_version().decode(),
            'opencv_build_sha256': hashlib.sha256(lib.opencv_build_info()).hexdigest(),
            'bridge_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
