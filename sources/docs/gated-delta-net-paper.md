---
id: doc-gated-delta-net-paper
title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
author: Songlin Yang, Jan Kautz, and Ali Hatamizadeh
url: https://arxiv.org/abs/2412.06464
source_category: paper
architectures: []
tags: [gated-delta-net, linear-attention, attention]
retrieved_at: 2026-08-16
---

# Gated Delta Networks paper

The paper introduces the gated delta recurrence and a hardware-efficient chunkwise training algorithm. In arXiv v3, §3.1 Equation (10) is

`S_t = S_{t-1}(alpha_t(I - beta_t k_t k_t^T)) + beta_t v_t k_t^T`.

Equivalently, decay the old state by `alpha_t`, predict the current value from that decayed state, and apply the delta correction. The same recurrence is Equation (8) in arXiv v1; locators must therefore name the paper version.

This algorithmic source does not by itself establish an NVIDIA architecture, kernel implementation, or performance result. Architecture- and implementation-specific claims require separate code or benchmark evidence.
