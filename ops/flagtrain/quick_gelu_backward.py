# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NVIDIA baseline for quick_gelu (backward).

The benchmark's ``test_quick_gelu_backward_perf`` times only
``torch.autograd.grad``; the forward runs outside the timed region. This module
keeps the same convention:

* ``build_inputs`` runs one forward outside the timed region to get ``out`` and
  a random ``grad``;
* ``native()`` only calls ``torch.autograd.grad(out, x, grad)``.

So NCU profiles only the backward kernel and do_bench times only the backward.
The forward reference is Megatron's eager ``quick_gelu`` (``jit_fuser``
disabled). Self-contained module; depends only on torch and megatron. When
Megatron is unavailable it raises instead of falling back to a torch
composition.
"""

import torch

OP_NAME = "quick_gelu_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False

# benchmark/megatron/test_quick_gelu.py::_CORE_SHAPES
_SHAPES = [
    (1,),
    (1024,),
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (4, 8192, 4096),
    (8, 4096, 11008),
]


def base_op():
    """Megatron-LM quick_gelu (eager); raises when Megatron is unavailable."""
    try:
        import megatron.core.jit as megatron_jit

        # Megatron's @jit_fuser is torch.compile by default. The collector must
        # profile Megatron's eager backward rather than Inductor output, so
        # disable the fuser before activations binds the decorator.
        megatron_jit.disable_jit_fuser()
        import megatron.core.activations

        kernel = megatron.core.activations.quick_gelu
        # activations may have been imported before the fuser was disabled; in
        # that case unwrap the torch.compile wrapper instead of compiling here.
        return getattr(kernel, "_torchdynamo_orig_callable", kernel)
    except Exception as exc:
        raise RuntimeError(
            "quick_gelu_backward baseline requires Megatron-LM's "
            "megatron.core.activations.quick_gelu, which could not be "
            f"imported: {exc!r}. Install megatron-core on this machine."
        ) from exc


def native():
    # The backward baseline is Megatron's quick_gelu forward autograd node;
    # check Megatron here so an unavailable install fails at native() time.
    base_op()

    def backward(out, x, grad_output):
        return torch.autograd.grad(out, x, grad_output, retain_graph=True)

    return backward


def grid():
    return [{"shape": shape} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    # Forward runs here, outside the profiled/timed region.
    x = torch.randn(binding["shape"], dtype=dtype, device=device, requires_grad=True)
    out = base_op()(x)
    grad_output = torch.randn_like(out)
    return (out, x, grad_output), {}


def key_shape(binding):
    # The benchmark's recorded shape is the forward input x.
    return list(binding["shape"])
