# provenance: derived from blog-gated-delta-net, pr-sglang-21019; not upstream code
# origin: wiki/kernels/gated-delta-net.md Phase 3 variant (copied from extracted blog bundle)

# Extracted from sources/blogs/gated-delta-net.md by scripts/extract_blog_code.py
# Heading: ## Key Code > ### Triton decode-step kernel (streaming)
# Original fence language: python
# See artifacts/blogs/gated-delta-net/code/PROVENANCE.yaml for origin + license metadata.

import triton
import triton.language as tl

@triton.jit
def gdn_decode_step_kernel(
    Q, K, V, GATE, STATE, OUT,
    stride_qb, stride_kb, stride_vb,
    Dk: tl.constexpr, Dv: tl.constexpr):
    """
    One-token delta-rule update for decode. STATE is a [Dk, Dv] matrix kept
    per sample; we fold in the new (k,v) pair after applying the decay gate.
    """
    b = tl.program_id(0)
    dk = tl.arange(0, Dk)
    dv = tl.arange(0, Dv)

    q = tl.load(Q + b * stride_qb + dk)                          # [Dk]
    k = tl.load(K + b * stride_kb + dk)                          # [Dk]
    v = tl.load(V + b * stride_vb + dv)                          # [Dv]
    g = tl.load(GATE + b)                                        # scalar decay

    state = tl.load(STATE + b * Dk * Dv + dk[:, None] * Dv + dv[None, :])
    state = state * g                                             # apply decay
    state = state + k[:, None] * v[None, :]                       # delta update
    tl.store(STATE + b * Dk * Dv + dk[:, None] * Dv + dv[None, :], state)

    out = tl.sum(q[:, None] * state, axis=0)                      # [Dv]
    tl.store(OUT + b * Dv + dv, out)
