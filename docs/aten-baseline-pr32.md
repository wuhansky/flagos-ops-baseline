# ATen baseline location for PR #32

Native CUDA measurements from this PR are stored in `op_pytorch_h800_perf_baseline.json`.
This is an offline data relocation: 196 original observations from
4 source artifacts, without new timing or numerical validation.
Every record/configuration is unchanged. Historical main-file records and
collection headers are unchanged. This commit changes no producer or timer code.

Imported shape keys use `pr<number>:<original artifact stem>:<original shape key>`
to preserve different input contracts, overloads and collection runs. Source
information is under `_aten_consolidation.sources`: exact original PR head,
artifact Git blob/SHA256, original metadata, per-operator entry metadata and
`shape_keys` mapping original keys to new keys. Adjacent provenance is also
embedded with original hashes and raw JSON text. Existing provenance sidecars
remain historical collection evidence and are not rewritten.

| Source ID | Imported operator keys |
|---|---|
| `pr32:op_inspect216_h800_inplace_baseline` | `addmm_`, `baddbmm_` |
| `pr32:op_inspect216_h800_perf_baseline` | `_flash_attention_forward`, `pack_padded_sequence`, `quantized_max_pool2d`, `quantized_max_pool2d_out` |
| `pr32:op_inspect216_h800_sdpa_baseline` | `scaled_dot_product_flash_attention` |
| `pr32:op_inspect216_h800_sdpa_kunlun_baseline` | `_scaled_dot_product_flash_attention` |

Select the intended source, operator, original shape key and dtype explicitly:

```python
source = baseline['_aten_consolidation']['sources'][source_id]
shape_key = source['operators'][operator]['shape_keys'][original_shape_key]
record = baseline[operator]['shapes'][shape_key][dtype]
```

Continue to check the full original input/overload/timing contract. Do not select
by shape alone or use the main historical environment for an imported record.
Original metadata may describe entries excluded from a mixed collection; only
the operators explicitly indexed above were imported. Source observations are
not a count of deduplicated workloads. This snapshot is not a resume checkpoint.

Original standalone snapshots remain recoverable from immutable source commit
`5827400565112103e8bb06462c87f9caf1a76fd7`. Historical collection notes and sidecars refer to those original
names/hashes. A FlagGems consumer pinned to an original artifact digest must be
migrated explicitly; changing its filename alone is insufficient. For unchanged
historical consumers, retrieve the exact original bytes via `git show
5827400565112103e8bb06462c87f9caf1a76fd7:<original artifact path>`.

Non-ATen entries, host/CPU artifacts and mixed source files containing them stay
unchanged. Retained mixed artifacts: `op_inspect216_h800_perf_baseline.json`.

Validation: all selected operator entries reconstruct exactly from their pinned
source blobs; all historical data/metadata are preserved; only this PR's native
sources were imported; shape destinations are unique. File size:
92,475,558 bytes. SHA256: `6a6277aa9e2baab389fe3e3a3377267801cc6fc20ba9eec76f7617d9dbf25548`.

Exact original snapshot bytes are embedded in each source's `raw_json`, and the
profile manifest has the same field under `auxiliary_artifacts`. This preserves
existing digest-based readers without guessing JSON serialization. Mixed-source
raw evidence retains its historical complete contents; only the selected native
entries are exposed as imported operator records. A reader must verify the
original digest and compare those entries with the indexed unified records.
