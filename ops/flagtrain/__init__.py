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

"""FlagTrain 基线采集模块（方案 B）。

这些模块采集的是 FlagTrain benchmark 在 NVIDIA 上使用的**基线实现**，而不是
FlagTrain 自己的 Triton 实现。使用方式与 ops/vllm、ops/sglang 相同：

    python tools/collect_baseline_nvidia.py --repo flagtrain --ops lamb ...

模块契约见 ops/vllm/__init__.py 与 tools/_FORK_CONTRACT.md：

    OP_NAME / DTYPES / IS_INPLACE
    native() -> callable | None
    grid() -> list[dict]
    build_inputs(binding, dtype, device) -> (args, kwargs)
    key_shape(binding) -> list[int] | str
    config(binding, dtype) -> dict        # 可选
"""
