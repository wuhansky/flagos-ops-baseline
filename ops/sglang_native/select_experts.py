# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import _sglang_profiles
import torch

OP_NAME = "select_experts"
DTYPES = [torch.bfloat16]
BENCH_RETURN_MODE = "median"
IS_INPLACE = False


def native():
    return _sglang_profiles.native(OP_NAME)


def grid():
    return _sglang_profiles.grid(OP_NAME)


def build_inputs(binding, dtype, device):
    return _sglang_profiles.inputs(OP_NAME, binding, dtype, device)


def key_shape(binding):
    return _sglang_profiles.key_shape(OP_NAME, binding)


def config(binding, dtype):
    return _sglang_profiles.config(OP_NAME, binding, dtype)


def make_input_reset(**inputs):
    return _sglang_profiles.make_input_reset(OP_NAME, inputs)


def time_with_reset(fn, reset):
    return _sglang_profiles.time_with_reset(fn, reset)


def timing_contract():
    return _sglang_profiles.timing_contract()
