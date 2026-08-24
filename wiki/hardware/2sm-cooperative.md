---
id: hw-2sm-cooperative
title: "Two-CTA Cooperative MMA"
type: hardware
architectures: [sm100, sm100a]
tags: [2sm-cooperative, tcgen05, cluster]
confidence: source-reported
related: [hw-tcgen05-mma, hw-tmem, technique-warp-specialization]
sources: [doc-ptx-isa-sm100, doc-cutlass-cute-dsl, blog-colfax-cutlass]
aliases: ["2-SM cooperative", "dual CTA", "2CTA", "cta_group::2"]
---

# Two-CTA cooperative MMA

`tcgen05` operations with `cta_group::2` act on a CTA pair. One thread from the pair can initiate an MMA while the peer CTA is active; the operation touches TMEM belonging to both CTAs. NVIDIA uses **TPC** to mean **Texture Processing Cluster**, not “Two Processing Clusters.”

```ptx
tcgen05.mma.cta_group::2.kind::f16
    [d_tmem], a_desc, b_desc, idesc, enable_input_d;
```

## Correctness constraints

- The CTAs must form a valid CTA pair and the peer must remain active when the operation is issued.
- All `tcgen05` instructions in the kernel must use the same CTA-group value.
- Allocation management for `cta_group::2` is collective across one fully active warp in each peer CTA; the first warp may block until its peer executes the matching operation.
- Paired TMEM and shared-memory operands must follow the selected instruction’s addressing and layout tables.
- Cross-CTA synchronization and lifetime rules must be satisfied before a peer exits or its memory is reused.

Whether the paired form is faster is workload- and configuration-dependent. Compare it with a one-CTA form using the same datatype, problem, layout, clock conditions, and reporting convention.
