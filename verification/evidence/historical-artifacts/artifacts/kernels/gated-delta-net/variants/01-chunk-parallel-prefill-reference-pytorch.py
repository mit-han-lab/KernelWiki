# provenance: derived from blog-gated-delta-net, pr-sglang-21019; not upstream code
# origin: wiki/kernels/gated-delta-net.md Phase 3 variant (copied from extracted blog bundle)

# Extracted from sources/blogs/gated-delta-net.md by scripts/extract_blog_code.py
# Heading: ## Key Code > ### Chunk-parallel prefill reference (PyTorch)
# Original fence language: python
# See artifacts/blogs/gated-delta-net/code/PROVENANCE.yaml for origin + license metadata.

import torch

def gated_delta_net_prefill(q, k, v, gate, initial_state, CHUNK_SIZE=64):
    """
    Chunk-parallel prefill. Each chunk's state matrix is reused across its
    query window, so we pay the O(Dk*Dv) state update once per chunk, not
    per token.
    q, k: [B, L, Dk]    v: [B, L, Dv]    gate: [B, L]
    """
    B, L, Dk = q.shape
    Dv = v.shape[-1]
    out = torch.empty(B, L, Dv, device=q.device, dtype=q.dtype)
    state = initial_state.clone()                  # [B, Dk, Dv]
    for ci in range(0, L, CHUNK_SIZE):
        ce = min(ci + CHUNK_SIZE, L)
        k_chunk = k[:, ci:ce]
        v_chunk = v[:, ci:ce]
        g_chunk = gate[:, ci:ce]
        decay = torch.cumprod(g_chunk, dim=1)       # adaptive memory decay
        for t in range(ce - ci):
            state = state * decay[:, t:t+1, None]
            state = state + k_chunk[:, t, :, None] * v_chunk[:, t, None, :]
            out[:, ci + t] = (q[:, ci + t, :, None] * state).sum(dim=1)
    return out, state
