---
id: hw-mbarrier
title: "mbarrier (Memory Barrier Primitives)"
type: hardware
architectures: [sm100, sm100a, sm90, sm90a, sm80]
tags: [mbarrier, tma]
confidence: source-reported
related: [hw-tma, hw-tcgen05-mma, technique-warp-specialization, technique-pipeline-stages]
sources: [doc-ptx-isa-sm100]
aliases: [mbarrier, "memory barrier", "mbar"]
blackwell_relevance: "Transaction-count mbarriers coordinate asynchronous TMA, CLC, and tcgen05 completion with consuming threads."
---

# mbarrier

An mbarrier is a shared-memory barrier object with phase tracking. The primitive was introduced for `sm_80` in PTX ISA 7.0. Hopper-era PTX added transaction-count operations used by asynchronous bulk copies; Blackwell operations such as CLC and `tcgen05.commit` also use mbarrier completion mechanisms.

## Two accounting dimensions

- **Arrival count:** participating threads or agents arrive until the pending count reaches zero.
- **Transaction count:** `expect_tx` adds expected asynchronous bytes; a completing asynchronous operation reduces that count.

The phase completes only when both forms of pending work satisfy the instruction’s rules. Reusing an object requires tracking its phase/parity and respecting invalidation and memory-ordering requirements.

```ptx
mbarrier.init.shared::cta.b64 [bar], arrivals;
mbarrier.arrive.expect_tx.shared::cta.b64 state, [bar], expected_bytes;
// issue the asynchronous operation that names [bar]
wait:
mbarrier.try_wait.parity.shared::cta.b64 ready, [bar], phase;
@!ready bra wait;
```

Exact qualifiers vary with scope and PTX version. An asynchronous engine’s completion transaction must not be replaced by an extra manual arrival. Likewise, an `expect_tx` call is not itself proof that the associated copy was issued or completed.
