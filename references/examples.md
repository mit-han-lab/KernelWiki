---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Worked query examples

These examples show evidence-preserving navigation. Commands locate candidates; they do not prove factual correctness by themselves.

## 1. Find B200 GEMM material

```bash
python3 scripts/query.py --type kernel --tag gemm --architecture sm100
python3 scripts/get_page.py kernel-nvfp4-gemm --follow-sources
```

Read the kernel contract first, then its PTX/CUTLASS sources. Do not reuse a tutorial's stage count, warp map, or throughput without matching its configuration.

## 2. Diagnose low SM utilization

```bash
python3 scripts/query.py --symptom low-sm-utilization
python3 scripts/get_page.py pattern-low-sm-utilization
```

Use the pattern page to choose measurements. CLC, persistent scheduling, and tile-shape changes are candidates, not automatic fixes.

## 3. Locate CUTLASS `tcgen05` evidence

```bash
python3 scripts/query.py --tag tcgen05 --repo cutlass --limit 30
python3 scripts/grep_wiki.py "tcgen05\\.mma" --only sources
```

Open the pinned PR or official CUTLASS file before quoting an instruction kind or role allocation.

## 4. Inspect FlashAttention-4

```bash
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
```

Keep the paper's full sweep and maximum-result boundary. Do not assign its end-to-end gain to partial software exponential or ping-pong scheduling alone.

## 5. Compare Hopper WGMMA with Blackwell `tcgen05`

```bash
python3 scripts/get_page.py migration-wgmma-to-tcgen05
python3 scripts/get_page.py hw-tcgen05-mma
```

Compare execution ownership, accumulator storage, descriptors, synchronization, and legal shapes; mnemonic replacement alone is insufficient.

## 6. Research the GPU Mode NVFP4 tasks

```bash
python3 scripts/query.py --type contest --tag nvfp4
python3 scripts/get_page.py contest-gpumode-p1
```

The public repository defines task contracts but not a complete podium. Yue and Amandeep provide author-reported Problem 1 measurements; unavailable/private rankings must remain unverified.

## 7. Research Gated Delta Net on Blackwell

```bash
python3 scripts/query.py "gated delta net decode" --architecture sm100
python3 scripts/get_page.py kernel-gated-delta-net --follow-sources
```

Separate the mathematical recurrence, framework interface, and actual GPU implementation. A simplified recurrence sketch is not a production Triton kernel.

## 8. Investigate a memory-bound kernel

```bash
python3 scripts/query.py --symptom memory-bound
python3 scripts/get_page.py technique-vectorized-loads --follow-sources
python3 scripts/get_page.py technique-cache-policy --follow-sources
```

Check alignment, instruction width, cache reuse, occupancy, and measured bandwidth. Cache qualifiers and wider loads are workload-dependent.

## 9. Find FlashInfer FP8 MoE PRs

```bash
python3 scripts/query.py --repo flashinfer --tag moe --limit 30
python3 scripts/query.py --repo flashinfer --tag fp8 --limit 30
```

PR frontmatter is an intake index. Open the PR and pinned artifact before treating a tag or summary as semantic proof.

## 10. Inspect SM100 PTX

```bash
python3 scripts/get_page.py lang-ptx --body-only
python3 scripts/get_page.py hw-tmem --follow-sources
```

Use the official PTX ISA for exact syntax and constraints. Wiki snippets may be logical references or scoped examples; the validator checks structure, not compilation.

## Synthesis pattern

1. Frame the topic with a wiki page.
2. Follow its source IDs to primary evidence.
3. State architecture, version, dtype, shape, and operation boundaries.
4. Label code as upstream, executable local code, logical reference, or pseudocode.
5. Quote performance only with the source's benchmark context and limitations.

Avoid treating generated indices, candidate tags, validator success, or absence of contradictory evidence as factual proof.
