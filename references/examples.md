---
version_sensitive:
  id: vs-triton-3.6-blackwell-tcgen05
---

# Worked Query Examples

Concrete patterns showing how to translate a user question into a navigation path and synthesize an answer.

## Task-specific retrieval and reading

Start with the [strategy workflow](search-strategies.md). During optimization,
the calling LLM updates context with measurements, prior reads and experiment
outcomes, then turns that evidence into queries and candidate priorities. Select
keyword, multi-query or guided directly; `query.py` executes the decision using
the same corpus, aliases and filters, within the caller's output budget.

The `keyword` commands below cover focused lookups with the original scoring;
`master` remains a compatible alias. Conditional examples alongside them show
multi-query for complementary hypotheses and guided for new measurements.
Select directly from the current evidence; no preset is a required first step.
The measurement scenarios are illustrative, not experimental results. JSON context
and custom Python are optional; see the [preset reference](search-strategies.md#start-a-run).

The commands below run from the wiki root. Set `search_run` to an absolute path
in the optimization workspace first:

```bash
search_run="<optimization-directory>/tmp/kernelwiki-search"
```

Read promising pages and follow their sources:

```bash
python3 scripts/get_page.py pattern-pipeline-stalls --follow-sources
python3 scripts/get_page.py hw-mbarrier
```

For an experiment, add `--run-dir <run-dir> --query-id <query-id>` to each read.
Source previews contain only the first 500 body characters; read the source
itself when that preview is insufficient. Use supplementary literal or regex
search to locate code, then fetch the relevant artifact path with `get_page.py`
when the returned code needs to be recorded.

```bash
rg -l -F 'mbarrier' artifacts/prs/cutlass
```

For ablations, fix corpus, model, tasks, hardware and total research budget.
Record the generated strategy, context, emitted and omitted candidates, actual
page/code output, and resulting kernel timings. The receipts describe tool
activity; they do not measure model attention or prove selection quality. Compare
with a competent master workflow that can reformulate queries and follow links.
Distinguish gains from new candidate coverage, richer excerpts and LLM selection.
Count useful evidence and its effect on kernel changes, not just calls to each
strategy: multiple master queries can also explore different mechanisms.

---

## Example 1: "How do I write a fast GEMM kernel on B200?"

**Navigation path**:
1. `queries/by-kernel-type.md` → find `gemm` row → lists kernel pages
2. Read `wiki/kernels/fp8-block-scale-gemm.md` and `wiki/kernels/nvfp4-gemm.md`
3. Follow `sources:` to concrete PRs: `pr-cutlass-2139` (blockwise+groupwise GEMM), `pr-vllm-13798` (FP8 GEMM)
4. For optimization path, read `wiki/techniques/warp-specialization.md`, `wiki/techniques/persistent-kernels.md`
5. For progression, cite the tcgen05 tutorial (`sources/blogs/tcgen05-tutorial.md`): Naive 17% → 128B-swizzled 3D TMA 46% → Pipelining 62% → Warp specialization 80% → 2-SM 86% → persistent static scheduling 98%

**Command**: a focused GEMM lookup.

```bash
python3 scripts/query.py "GEMM" --strategy keyword --type kernel --architecture sm100 --run-dir "$search_run" --compact
```

If the bottleneck is unclear, compare tile efficiency, loading and synchronization
as separate hypotheses. Avoid a kernel-only filter when mechanism pages matter:

```bash
python3 scripts/query.py --strategy multi-query --query "GEMM tile" --query "TMA layout" --query "mbarrier wait" --run-dir "$search_run" --compact
```

If a trace instead identifies producer waits as the current issue, focus the GEMM
pool on pipeline mechanisms. This needs neither multiple queries nor custom code:

```bash
python3 scripts/query.py "GEMM" --strategy guided --focus "pipeline producer" --reason "Illustrative: trace identifies producer waits" --run-dir "$search_run" --compact
```

---

## Example 2: "My kernel has low SM utilization on B200"

**Navigation path**:
1. `queries/by-problem.md` → find "low-sm-utilization" row
2. Pattern page: `wiki/patterns/low-sm-utilization.md`
3. Candidate techniques: `technique-persistent-kernels`, `technique-tile-scheduling`, `hw-clc`
4. Read the CLC page: `wiki/hardware/clc.md` for code example
5. Cite: the tcgen05 tutorial showed 86% → 98% with static persistent scheduling. It presents CLC only as a possible follow-on exercise; use `wiki/hardware/clc.md` for the separate CLC mechanism.

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --symptom low-sm-utilization
python3 scripts/get_page.py pattern-low-sm-utilization
```

---

## Example 3: "Where is tcgen05.mma used in CUTLASS?"

**Navigation path**:
1. `queries/by-hardware-feature.md` → find `tcgen05` row → lists all relevant pages
2. `queries/by-repo.md` → NVIDIA/cutlass section
3. Cross-reference via tag filter

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --tag tcgen05 --repo cutlass --limit 30
python3 scripts/grep_wiki.py "tcgen05\\.mma" --only sources
```

Tip: the `--tag` filter also accepts aliases, so `--tag UMMA` resolves to `tcgen05`.

---

## Example 4: "Show me the FlashAttention-4 implementation details"

**Navigation path**:
1. Direct: `wiki/kernels/flash-attention-4.md`
2. Performance: up to 1613 TFLOPS on B200 BF16 (71%) in the paper's sweep
3. Techniques used: ping-pong scheduling, software exp, 2-CTA backward
4. Follow sources → `sources/docs/flash-attention-4.md` (paper), `sources/blogs/flash-attention-4.md` (author blog)

**Command**:
```bash
python3 scripts/get_page.py kernel-flash-attention-4 --follow-sources
```

---

## Example 5: "What's the difference between Hopper wgmma and Blackwell tcgen05?"

**Navigation path**:
1. `wiki/migration/wgmma-to-tcgen05.md` — dedicated migration guide with `blackwell_relevance` field
2. `wiki/hardware/tcgen05-mma.md` — canonical reference for the new instruction
3. Contrast with Hopper behavior implicit in the migration page

**Command**:
```bash
python3 scripts/get_page.py migration-wgmma-to-tcgen05
python3 scripts/get_page.py hw-tcgen05-mma
```

---

## Example 6: "How did teams in the GPU Mode NVFP4 Hackathon win?"

**Navigation path**:
1. `sources/contests/gpu-mode-nvfp4/` (4 problems)
2. Each problem page records the organizer task definition, speed-of-light values when published, and a dated leaderboard snapshot; no page declares a per-submission listing
3. Read participant blogs: `sources/blogs/yue-nvfp4-hackathon.md`, `sources/blogs/amandeep-nvfp4-attempts.md`, `sources/blogs/simon-nvfp4-gemv.md`
4. Techniques: PTX-level control, cache policy differentiation, register budgeting

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --type contest --tag nvfp4
python3 scripts/get_page.py contest-gpumode-p1
```

---

## Example 7: "Write a Triton kernel for GatedDeltaNet decode on Blackwell"

**Navigation path**:
1. `wiki/kernels/gated-delta-net.md` — conceptual + code
2. `wiki/languages/triton-blackwell.md` — current Triton 3.6+ Blackwell lowering surfaces (tcgen05 + TMEM via descriptor/TMA + warp_specialize, `tl.dot_scaled`, Gluon multi-CTA); its opening paragraph records that the earlier pre-3.6 no-TMEM/no-tcgen05 framing is obsolete for 3.6
3. Source PRs: `pr-vllm-*` for gated_delta, FlashInfer GDN kernels

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" "GDN decode" --language triton
```

After correctness is established, suppose profiling shows excessive intermediate
memory traffic. Change priorities and add a complementary query to reach material
that may not mention GDN. Check each returned page against the actual workload:

```bash
python3 scripts/query.py "GDN" --strategy guided --focus "fusion reuse" --extra-query "intermediate memory" --language triton --reason "Illustrative: intermediate memory traffic is now the bottleneck" --run-dir "$search_run" --compact
```

---

## Example 8: "What are the memory-bound kernel optimization tricks on B200?"

**Navigation path**:
1. `wiki/patterns/memory-bound.md` — candidate techniques list
2. `wiki/techniques/vectorized-loads.md` — wide loads + cache policies (covers `evict_first` / `no_allocate`)
3. Case study: `wiki/kernels/nvfp4-gemv.md` records the author's 22.392 μs final aggregate; the staged 2000 → 443 → 39 → 27 → 22.392 μs sequence is on `wiki/techniques/vectorized-loads.md`, where the source attributes each step to a combination of edits

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --symptom memory-bound
```

If the measurement has not distinguished transaction efficiency from cache reuse,
query those mechanisms separately before choosing an experiment:

```bash
python3 scripts/query.py --strategy multi-query --query "vectorized loads" --query "cache policy" --query "memory coalescing" --run-dir "$search_run" --compact
```

---

## Example 9: "Find all FlashInfer PRs for FP8 MoE"

**Command**:
```bash
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --repo flashinfer --tag moe --limit 30
python3 scripts/query.py --strategy keyword --run-dir "$search_run" --repo flashinfer --tag fp8 --limit 30
python3 scripts/grep_wiki.py "fp8" "moe" --only sources --files-only
```

---

## Example 10: "What PTX instructions are unique to SM100?"

**Navigation path**:
1. `wiki/languages/ptx-sm100.md` — direct reference
2. Cross-reference: tcgen05.alloc/mma/ld/st/dealloc/fence, clusterlaunchcontrol.try_cancel, cp.async.bulk.tensor multicast

**Command**:
```bash
python3 scripts/get_page.py lang-ptx --body-only
python3 scripts/grep_wiki.py "tcgen05" --only wiki --context 0
```

---

## Synthesis Pattern

For most questions, a high-quality answer follows this shape:

```
1. Topic framing (1-2 sentences, cite wiki page)
   → Pull from wiki/<section>/<topic>.md ## Overview

2. Technical mechanism (cite hardware + technique pages)
   → wiki/hardware/*.md for hw details
   → wiki/techniques/*.md for optimization patterns

3. Concrete code snippet (copy from wiki page, already validated)
   → Every technique/kernel/language page has a compilable snippet

4. Performance evidence (cite performance_claims)
   → Always report gpu, dtype, shape, metric, value, source_id

5. References (cite source IDs)
   → PR: pr-cutlass-2139 → sources/prs/cutlass/PR-2139.md
   → Blog/doc: blog-* / doc-*
```

## Anti-Patterns (Don't Do These)

- Don't recommend techniques without citing `sources:` — the wiki exists precisely for this.
- Don't quote performance without the full 6-field `performance_claims` record.
- Don't conflate `sm90` and `sm100` patterns — always check the `architectures:` field.
- Don't cite `verified` claims without checking the page actually has `evidence_basis` entries that name both an official doc and an upstream code source.
- Don't recommend DeepEP/DualPipe/EPLB — they're explicitly out of scope (kernel-only KB).
