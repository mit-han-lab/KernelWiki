# Triton Blackwell Version Evidence

## Releases of record

| Release | Exact revision | Evidence-scoped conclusion |
|---|---|---|
| v3.2.0 | `9641643da6c52000c807b5eeed05edaec4402a67` | Checked negative side: the corresponding TCGen5 MMA, TMEM, and MMAv5-lowering symbols are absent. |
| v3.3.0 | `819e9c8c29ad2ae96cbd93a1d3b8a3a0f4c8f09c` | Adds TCGen5 MMA/scaled-MMA operations, TMEM operations and allocation, MMAv5 lowering, and conversion tests for concrete TCGen5/TMEM output. |
| v3.5.0 | `c3c476f357f1e9768ea4e45aa5c17528449ab9ef` | Includes explicit Gluon TCGen5/TMEM and Blackwell block-scaled matmul tutorials, plus documented warp-specialization work. |
| v3.6.0 | `7c56a5e40f7fd928dfd5c72902d5def0097db73a` | Incremental generalization of copies/layouts and MMA, aref warp-specialization work, and initial multi-CTA/2-CTA Gluon support. |
| v3.7.0 | `5f3f125e8f63c24613f1f73b937442864f263f94` | Further end-to-end 2-CTA, multicast, and TMA work. |
| v3.7.1 | `f797708` | Latest checked stable release on 2026-08-08; a two-regression patch with no advertised new API or feature. |

The discriminating version boundary is therefore v3.2.0 to v3.3.0, not v3.5.x to v3.6.0. See `sources/docs/triton-3.3-blackwell.md` for the pinned files and scope.

## What the compiler evidence proves

The exact v3.3.0 tree proves that native Blackwell TCGen5/TMEM compiler machinery and tested lowering exist. The v3.5.0 tutorials prove explicit user-visible examples for Gluon TCGen5/TMEM and `tl.dot_scaled` block-scaled matmul by that tag. The v3.6.0 notes prove additional Blackwell work, not first introduction.

None of those facts implies that arbitrary plain `tl.dot` kernels select TCGen5 for every dtype, layout, shape, architecture, or compiler configuration. A target-specific selection claim needs inspectable IR or PTX from that configuration.

## Downstream evidence and limits

- `pr-vllm-34597`, pinned at merge SHA `a1257fd1`, adds FP8 KV-cache handling to vLLM's generic Triton MLA decode backend. Its kernel contains `tl.dot`; neither the PR nor exact code supplies an SM100 guard, a Triton-version pin, a TCGen5/TMEM symbol, or emitted PTX. The PR specifically motivates the backend as the MLA option available on SM120.
- `pr-vllm-29339`, pinned at `c17610e2`, gates MXFP4 `triton_kernels` dispatch to SM90 and SM100. It changes dispatch code, not a Triton kernel or compiler lowering.
- `pr-sglang-21019`, pinned at `5bdc07d9`, contains a Triton GatedDeltaNet projection rearrangement using loads and stores, with no `tl.dot`.
- `pr-sglang-22079`, pinned at `5638d40f`, contains an extend-attention Triton kernel with real `tl.dot` operations. It still contains no emitted-PTX witness for a particular MMA instruction.

These artifacts verify downstream Triton use. They do not independently verify which backend instruction any target selects.

## Scoped ecosystem observations

SGLang PR 5390 reports one DeepSeek-R1 MLA benchmark in which the CUTLASS run records 10,447.34 total tok/s and the Triton run 8,227.35 total tok/s, a 26.98% difference under the PR's recorded 3,000-prompt, TP8/DP8, float16, 1,000-input/1,000-output-token scope. It is not a universal language comparison.

SGLang PR 21595 changes the SM100 datacenter multimodal-attention default from `triton_attn` to FA4. This is a scoped routing decision, not evidence about every Blackwell workload.

The live FlashInfer-Bench author leaderboard retrieved 2026-08-08 reports Gemini 2.5 Pro at 0.628x and 73.1% resolved, GPT-5 at 0.467x and 92.3%, and Claude Opus 4.1 at 0.456x and 73.1%, each over 660 workloads. The page does not attach those rows to a Triton version or a Triton-only language subset.

## Open proof obligation

The checked downstream bundles contain no explicit TCGen5 PTX dump or warp-specialized descriptor/TMA lowering record. Until such evidence is captured for a specific kernel, architecture, dtype, shape, and toolchain, the wiki must not infer exact instruction selection from `tl.dot` source alone.
