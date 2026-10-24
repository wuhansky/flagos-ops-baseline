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

"""FlagTrain baseline-collection modules (option B).

These modules collect the **baseline implementations** the FlagTrain benchmark
uses on NVIDIA, not FlagTrain's own Triton kernels. They are driven like
ops/vllm and ops/sglang:

    python tools/collect_baseline_nvidia.py --repo flagtrain --ops lamb ...

The module contract is documented in ops/vllm/__init__.py and
tools/_FORK_CONTRACT.md:

    OP_NAME / DTYPES / IS_INPLACE
    native() -> callable
    grid() -> list[dict]
    build_inputs(binding, dtype, device) -> (args, kwargs)
    key_shape(binding) -> list[int] | str
    config(binding, dtype) -> dict        # optional
"""
