# Pinned SGLang native baselines

`ops/sglang_native` adds seven native workload profiles with 28 documented BF16
cases: Gemma RMSNorm (12), Gemma fused-add RMSNorm (4), optional-out SiLU (4),
TopK selection (2), the MoE kernel sequence (2), chunk GDN (2), and packed GDN
decode (2). Existing `ops/sglang` recipes remain independent.

These profiles require the companion FlagGems-sglang workload checkout, including
its pinned SGLang v0.5.11 environment (upstream commit
`612785ffdcaf35552f1ed433a981d596ca9fe900`). Follow that checkout's
`docs/operator_workloads.md` for offline wheel setup and validation. Missing source,
wrong package versions, changed entrypoints or unavailable FlashInfer fail rather
than selecting a Torch substitute.

```sh
export FLAGGEMS_SGLANG_SOURCE=/workspace/sglang-suite
python tools/collect_baseline_nvidia.py --repo sglang_native --device 0 \
  --reference-chip H800 --output op_sglang_native_h800_perf_baseline.json \
  --report-dir sglang-native-ncu
```

The functions selected by the native adapter include CUDA/FlashInfer and Triton
implementations. TopK's native configuration is prepared outside the timer.
Read-only profiles retain median `triton.testing.do_bench` latency. Read-write
profiles use the companion checkout's shared `measure_with_reset`: every call
starts from the original values, and device events bracket only the native
operator. Restoration plus a 256 MiB cache flush precede each event interval.
NCU warmups also restore the original input; restoration/cache preparation and
synchronization stay outside the target NVTX range. NCU cache control is `all`.
The same helper is used by target benchmarks. Other recipes without this
opt-in retain their existing timer and NCU argument-rebuild behavior.

Keys hash full tensor geometry (including stride and storage offset),
scalar/configuration ABI and timing mode. GDN pools retain their documented full
sizes. Configuration records contain input geometry, upstream revision and the
shared workload source SHA256. Collection `_environment` describes the actual
host/software/run; examine it alongside per-case NCU records before comparing.

Pass the resulting JSON through target FlagGems-sglang benchmark `--base-data`.
Only equal configuration keys should match; locally timed Torch reference-as-
candidate results validate the pipeline, not an optimized kernel.

## Published H800 snapshot

Both this snapshot (28 cases) and `op_sglang_mrotary_h800_perf_baseline.json`
(6 cases) were freshly collected on H800 after the timing fix. Ten
read-write keys and six mRoPE keys now include the reset-excluded timing
contract. The eighteen read-only records retain their timing contract; all 28 key
prefixes are normalized as described below. The original SGLang
implementations generated every latency/NCU record; no Torch timing substitute
was used. Seven target backends matched all 34 keys and timed their local
reference-as-candidate. These are pipeline checks, not optimized kernels.
Kunlunxin P800 comparisons retain raw ratios (`factor=None`) because its
memory-bandwidth factor is absent. See `validation/reset_timing_validation.json` for
exact collected source revisions, checks and retained evidence hashes.

`op_sglang_native_h800_perf_baseline.json` contains all seven interfaces and 28
core records collected on H800 with median latency and NCU counters. Each record
has a positive kernel count and retained raw `.ncu-rep` evidence. `_provenance`
binds the raw collection hash, exact remote/local source file hashes and Git
revisions, original native source revision, shared timing specification and complete
source-manifest hash.
The collector's Git fields remain null for the tar-staged remote checkout;
verified source provenance is recorded separately rather than inventing Git
state on that host.

The collector summary counts all seven public `sglang_fl_` operator keys,
including `sglang_fl_fused_moe_kernel_sequence`. The companion benchmark checks
all 28 configuration keys directly.

Golden data for these core cases was generated on H800 and uploaded to the
historical KS3 prefix recorded in the companion validation receipt. Ascend
replay of those same inputs passed 28/28 with no skips. This validates the portable Torch oracle;
it does not establish performance of a new backend candidate.

## Model-neutral names

Use the paired FlagGems-sglang checkout containing `operator_workloads.py` and
`verification/operator_common.py`. The snapshot was relabelled to `sglang-v1-*`
(read-only) / `sglang-v2-*` (read-write) without changing the geometry/ABI digest,
latency, counters, kernel records or scaling factors. Configuration labels now
record explicit dimensions. `_key_migration` records the original artifact hash;
original capture paths/source hashes remain verbatim in `_environment` and
`_provenance`. Receipts under `docs/validation/` are immutable historical evidence,
so their original filenames and hashes describe the pre-cleanup artifact.
No NVIDIA recollection or golden-tensor regeneration is implied by this rename.

The seven top-level baseline keys and collector `OP_NAME` values use the
`sglang_fl_` Test / Definition names. The 28 existing workload keys, latency
values, counters and collection metadata are preserved. Native entrypoints keep
their original names. Update the target FlagGems-sglang checkout and KGS baseline
selector together with these JSON files.
