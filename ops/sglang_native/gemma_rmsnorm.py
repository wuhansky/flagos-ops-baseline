# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import _sglang_profiles
import torch

OP_NAME = "sglang_fl_gemma_rmsnorm"
_NATIVE_OPERATOR = "gemma_rmsnorm"
DTYPES = [torch.bfloat16]
BENCH_RETURN_MODE = "median"
IS_INPLACE = False


def native():
    return _sglang_profiles.native(_NATIVE_OPERATOR)


def grid():
    return _sglang_profiles.grid(_NATIVE_OPERATOR)


def build_inputs(binding, dtype, device):
    return _sglang_profiles.inputs(_NATIVE_OPERATOR, binding, dtype, device)


def key_shape(binding):
    return _sglang_profiles.key_shape(_NATIVE_OPERATOR, binding)


def config(binding, dtype):
    return _sglang_profiles.config(_NATIVE_OPERATOR, binding, dtype)


def make_input_reset(**inputs):
    return _sglang_profiles.make_input_reset(_NATIVE_OPERATOR, inputs)


def time_with_reset(fn, reset):
    return _sglang_profiles.time_with_reset(fn, reset)


def timing_contract():
    return _sglang_profiles.timing_contract()
