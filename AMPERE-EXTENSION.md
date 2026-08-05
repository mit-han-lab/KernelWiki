# Ampere Extension (sm_80 / sm_86)

The knowledge base is Blackwell/Hopper-first. This extension makes the same pages actionable on Ampere hardware: A100 (sm_80) and GA10x consumer/workstation cards (sm_86: RTX 3090/3080, RTX A6000, A40). It was written against a 4x RTX 3090 rig, and numbers marked "measured" come from that machine.

## Design

All changes are **additive** (new files + vocabulary entries) so upstream merges stay trivial:

- `data/tags.yaml` — added architectures `sm80`, `sm86`; hardware features `cp-async`, `mma-sync`, `l2-persistence`.
- `data/aliases.yaml` — added alias groups for `sm80`, `sm86`, `cp-async`, `mma-sync`, `l2-persistence`.
- `sources/docs/` — 4 new source pages: `doc-ampere-tuning-guide`, `doc-ga102-whitepaper`, `doc-ptx-isa-ampere`, `doc-cutlass-ampere`.
- `wiki/hardware/` — 3 new pages: `hw-cp-async`, `hw-mma-sync-ampere`, `hw-ampere-memory-model`.
- `wiki/migration/` — 1 new page: `migration-hopper-to-ampere` (the entry point: instruction replacement table, capacity re-planning, scheduling paradigm, ncu checklist).
- `references/primer.md` — "Ampere Extension" section + alias cheat-sheet rows.
- `SKILL.md` — trigger description now includes Ampere/SM86/RTX 3090 and backport questions.

The upstream validator rules are untouched and still pass: Ampere pages don't trip the Blackwell-first rule (it only constrains Hopper-only pages), and every new page carries the full required frontmatter for its type.

## Query examples

```bash
python3 scripts/query.py --architecture sm86 --compact          # aliases: "RTX 3090", GA102 …
python3 scripts/query.py --tag cp-async --type hardware
python3 scripts/get_page.py migration-hopper-to-ampere
python3 scripts/grep_wiki.py "wait_group" --only wiki
```

## Verification status

Snippets on the new pages are compiled against `nvcc -arch=sm_86` on a real 4x RTX 3090 machine (see per-page `reproducibility`). Facts sourced from: NVIDIA Ampere tuning guide (CC 8.0/8.6 table), GA102 whitepaper v2.1 (tensor throughput incl. the FP32-accumulate half-rate), PTX ISA (instruction availability), CUTLASS (SM80 mainloop idiom).

## Candidate follow-ups (not done)

- `source-pr` pages for canonical Ampere kernels in tracked repos (e.g. vLLM Marlin W4A16, exllama kernels) — would upgrade wiki-page confidence to `verified` (needs an `upstream-code` source).
- `kernel-` case-study pages with benchmarked claims from the 4x3090 rig (FlashAttention-2, Marlin, Triton GEMM autotune points).
- sm_89 (Ada) column where it differs (FP8 tensor cores present, SMEM 100 KB like sm_86).
