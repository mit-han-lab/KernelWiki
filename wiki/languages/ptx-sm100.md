---
id: lang-ptx
title: "PTX for SM100"
type: language
tags: [ptx, tcgen05, tmem, tma, clc, mbarrier, nvfp4]
related: [hw-tcgen05-mma, hw-tmem, hw-clc, lang-cuda-cpp]
sources: [doc-ptx-isa-sm100]
reproducibility: snippet
architectures: [sm100, sm100a]
confidence: source-reported
---

# PTX for SM100

PTX is a versioned virtual ISA. A complete module declares both a PTX version and a target; support for one Blackwell feature does not imply support for every target-specific instruction.

This minimal module is useful as a compilation smoke test before adding an instruction under audit:

```ptx
.version 8.7
.target sm_100a
.address_size 64

.visible .entry no_op() {
  ret;
}
```

## Representative forms

```ptx
// Warp-collective allocation; destination is shared memory.
tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [dst], 32;

// Single-thread MMA issue; idesc is a required 32-bit instruction descriptor.
tcgen05.mma.cta_group::1.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;

// Track earlier MMA completion through an mbarrier.
tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [mma_done];

// Warp-collective deallocation before exit.
tcgen05.dealloc.cta_group::1.sync.aligned.b32 d_tmem, 32;
```

CLC is an asynchronous cancellation/acquisition protocol, not a one-operand “next tile” query:

```ptx
clusterlaunchcontrol.try_cancel.async.shared::cta
    .mbarrier::complete_tx::bytes.b128 [response], [done];
clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, response;
@p clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128
    {x, y, z, ignored}, response;
```

The exact legal kinds, shapes, layouts, address spaces, scopes, and target notes are normative in the current PTX ISA.
