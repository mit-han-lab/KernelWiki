# Query: By Repository

> Auto-generated. Do not edit manually.

<a id="dao-ailabflash-attention"></a>
## Dao-AILab/flash-attention
33 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#2497](../sources/prs/flash-attention/PR-2497.md) | [FA4][hd256] Backward TMA bulk-store epilogue + LSE/dpsum coalesce | 2026-04-27 |  | clc, tcgen05, tma |
| [#2488](../sources/prs/flash-attention/PR-2488.md) | [hd256] Improve forward kernel with exp2 FMA emulation (3% to 9% performance gain) | 2026-04-23 | pipeline-stages | attention, flash-attention, quantization |
| [#2489](../sources/prs/flash-attention/PR-2489.md) | [hd256] Add TMA paged KV support to SM100 2CTA forward kernel | 2026-04-23 |  | quantization, tma |
| [#2441](../sources/prs/flash-attention/PR-2441.md) | [Cute,Sm100,Fwd] add MLA 64/512 with topk sparsity for MQA 128 heads | 2026-04-06 | persistent-kernel, pipeline-stages | mla, tcgen05, topk |
| [#2412](../sources/prs/flash-attention/PR-2412.md) | Feat([FA4][CUTE DSL]) Add head_dim=256 support (forward + backward) | 2026-03-30 | persistent-kernel, pipeline-stages, swizzling | attention, clc, flash-attention |
| [#2360](../sources/prs/flash-attention/PR-2360.md) | [Fwd,Sm90] Add paged KV attention support (tma and cp.async) | 2026-03-17 | pipeline-stages | attention, mbarrier, tma |
| [#2333](../sources/prs/flash-attention/PR-2333.md) | Add SM120 varlen attention support | 2026-03-12 | tile-scheduling | attention |
| [#2218](../sources/prs/flash-attention/PR-2218.md) | [Ai-assisted] CLC work stealing | 2026-01-31 | pipeline-stages, swizzling, tile-scheduling | clc, mbarrier, tma |
| [#2186](../sources/prs/flash-attention/PR-2186.md) | [Cute,Fwd,Sm100] support irregular qhead / kvhead ratios | 2026-01-16 | pipeline-stages | attention, flash-attention, tma |
| [#2145](../sources/prs/flash-attention/PR-2145.md) | [CUTE][SM90]Enable pack-gqa with broadcasted maskmods | 2026-01-07 | pipeline-stages |  |
| [#2109](../sources/prs/flash-attention/PR-2109.md) | [Cute,Fwd,Sm100] fp8 e4m3 and e5m2 support | 2025-12-29 |  | attention, flash-attention, fp8 |
| [#2104](../sources/prs/flash-attention/PR-2104.md) | [Cute,Fwd,Sm100] distributed offset calculation for paged KV | 2025-12-28 |  | attention |
| [#2085](../sources/prs/flash-attention/PR-2085.md) | Add blocksparse support for bwd on blackwell | 2025-12-17 | pipeline-stages | attention, tma |
| [#2070](../sources/prs/flash-attention/PR-2070.md) | Add score-mod bwd support  | 2025-12-15 |  | tmem |
| [#2043](../sources/prs/flash-attention/PR-2043.md) | [Cute,Fwd] Extend score_mod to variable sequence length | 2025-12-03 |  |  |
| [#2033](../sources/prs/flash-attention/PR-2033.md) | [Cute,Bwd,Sm100] enable deterministic mode for sm100 bwd and fix race conditions | 2025-11-24 | tile-scheduling | tcgen05, tmem |
| [#2014](../sources/prs/flash-attention/PR-2014.md) | [Cute,Sm100,Fwd] use correction warps for epi when not using TMA | 2025-11-15 | tile-scheduling | tma |
| [#1999](../sources/prs/flash-attention/PR-1999.md) | [Cute,Fwd,Sm100] Support paged attention | 2025-11-11 | pipeline-stages | attention, flash-attention, tma |
| [#1985](../sources/prs/flash-attention/PR-1985.md) | [Cute] Block sparse support Sm100 | 2025-11-05 | pipeline-stages | attention, flash-attention, sparse-attention |
| [#1945](../sources/prs/flash-attention/PR-1945.md) | Blackwell FlashAttention-BWD (v1.0) | 2025-10-19 | pipeline-stages, swizzling, tile-scheduling | flash-attention, gemm, tcgen05 |
| [#1940](../sources/prs/flash-attention/PR-1940.md) | [Cute,Fwd,Sm100] Implement SplitKV | 2025-10-15 | persistent-kernel, tile-scheduling | attention, gemm, tcgen05 |
| [#1934](../sources/prs/flash-attention/PR-1934.md) | feat: Adding varlen support to cute-dsl sm80 bwd | 2025-10-13 | tile-scheduling | attention, flash-attention |
| [#1893](../sources/prs/flash-attention/PR-1893.md) | Improve causal backward determinism perf with SPT schedule | 2025-09-17 | swizzling, tile-scheduling |  |
| [#1823](../sources/prs/flash-attention/PR-1823.md) | Add sorting and head swizzle to varlen scheduler | 2025-08-19 | swizzling, tile-scheduling | sort |
| [#1604](../sources/prs/flash-attention/PR-1604.md) | Support hdimQK != hdimV backward | 2025-04-21 |  |  |
| [#1361](../sources/prs/flash-attention/PR-1361.md) | Fix FA3 Varlen Performance regression | 2024-11-29 | pipeline-stages | attention, flash-attention, tma |
| [#1331](../sources/prs/flash-attention/PR-1331.md) | FA3 paged attention: Readiness for Cutlass 3.6 / default value for block_table | 2024-11-12 |  | attention, flash-attention, gemm |
| [#1268](../sources/prs/flash-attention/PR-1268.md) | Paged Attention support for FA3 | 2024-10-10 |  | attention, flash-attention, fp8 |
| [#1236](../sources/prs/flash-attention/PR-1236.md) | FA3 kvcache + split kv + gqa parallelization | 2024-09-18 | tile-scheduling | fp8, gemm, tma |
| [#1233](../sources/prs/flash-attention/PR-1233.md) | Add local attention in Hopper FAv3 | 2024-09-16 |  | attention, fp8 |
| [#1173](../sources/prs/flash-attention/PR-1173.md) | FA3 FP8 qkv descales + restore max offset for h128 causal + added sync for producer WG | 2024-08-23 |  | fp8, gemm |
| [#1100](../sources/prs/flash-attention/PR-1100.md) | Fp8 kernel with "in-kernel" transpose of V in producer | 2024-07-26 | pipeline-stages, swizzling, tile-scheduling | attention, fp8, gemm |
| [#1072](../sources/prs/flash-attention/PR-1072.md) | Add var-seq-len to FA3 fp16 / bf16 fwd | 2024-07-19 |  | tma |

<a id="nvidiatensorrt-llm"></a>
## NVIDIA/TensorRT-LLM
88 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#14291](../sources/prs/TensorRT-LLM/PR-14291.md) | [None][feat] Update the logic of FMHA JIT path | 2026-05-19 | pipeline-stages, swizzling, tile-scheduling | attention, clc, flash-attention |
| [#14219](../sources/prs/TensorRT-LLM/PR-14219.md) | [None][feat] DSv4: enable GVR Heuristic Top-K for compress_ratio=4 | 2026-05-17 |  | attention, mla, reduction |
| [#14134](../sources/prs/TensorRT-LLM/PR-14134.md) | [None][feat] Add chunked prefill support for Gemma4 (text + vision multimodal) | 2026-05-14 |  | attention, moe |
| [#13975](../sources/prs/TensorRT-LLM/PR-13975.md) | [None][perf] Add CUDA q_b norm for DeepSeek V4 | 2026-05-11 |  |  |
| [#13929](../sources/prs/TensorRT-LLM/PR-13929.md) | [TRTLLM-35237][feat] Add cute dsl FP4 paged MQA logits decode kernel | 2026-05-09 |  | attention, fp4, fp8 |
| [#13938](../sources/prs/TensorRT-LLM/PR-13938.md) | [None][feat] Keep DSv4 o_a_proj as FP8, and port vLLM's fused_inv_rope_fp8_quant | 2026-05-09 | kernel-fusion | attention, block-scale, fp8 |
| [#13892](../sources/prs/TensorRT-LLM/PR-13892.md) | [None][perf] mHC fused_hc kernel optimizations + DS-V4 entry-boundary RMSNorm fold-in | 2026-05-08 | pipeline-stages, swizzling | attention, mbarrier, moe |
| [#13833](../sources/prs/TensorRT-LLM/PR-13833.md) | [None][perf] FC2 DenseGEMM autotune: split-K, swap_ab, fine-grained tuning buckets | 2026-05-07 | pipeline-stages, swizzling | fp4, gemm, moe |
| [#13811](../sources/prs/TensorRT-LLM/PR-13811.md) | [None][feat] Indexer topk opt | 2026-05-06 |  | sort, topk |
| [#13761](../sources/prs/TensorRT-LLM/PR-13761.md) | [None][perf] Optimize DeepSeek-V4 compressor BF16 input | 2026-05-05 |  |  |
| [#13767](../sources/prs/TensorRT-LLM/PR-13767.md) | [None][fix] Plumb swiglu_limit through DeepGEMM and TRTLLMGen FP8 fused MoE | 2026-05-05 |  | block-scale, fp8, moe |
| [#13771](../sources/prs/TensorRT-LLM/PR-13771.md) | [None][fix] Fix fused MHC for DeepSeek-V4-Pro hidden size | 2026-05-05 |  | tcgen05 |
| [#13740](../sources/prs/TensorRT-LLM/PR-13740.md) | [https://nvbugs/6108841][fix] add hidden_dim=6144 router GEMM instantiation for GLM-5 | 2026-05-04 |  | fp8, gemm, moe |
| [#13628](../sources/prs/TensorRT-LLM/PR-13628.md) | [None][feat] Fuse FP8 1x128 quantize + UE8M0 scale pack on SM100 | 2026-04-30 | kernel-fusion | attention, block-scale, fp8 |
| [#13630](../sources/prs/TensorRT-LLM/PR-13630.md) | [#13580][fix] AutoDeploy: Support Gemma3n/4 E2B variants | 2026-04-30 |  | attention, fp8 |
| [#13652](../sources/prs/TensorRT-LLM/PR-13652.md) | [None][feat] Add DeepSeekV4 attention kernels | 2026-04-30 |  | attention, flash-attention, mla |
| [#13477](../sources/prs/TensorRT-LLM/PR-13477.md) | [None][perf] Scheme X L2-aware dispatcher and PDL launchers for sparse-attention GVR Top-K | 2026-04-26 | pipeline-stages | attention, reduction, sort |
| [#13433](../sources/prs/TensorRT-LLM/PR-13433.md) | [None][perf] Extend customMoeRouting kernel to support Qwen3.5 | 2026-04-24 |  | fp4, moe, nvfp4 |
| [#13340](../sources/prs/TensorRT-LLM/PR-13340.md) | [None][feat] Integrate FP4 indexer for DSA on Blackwell | 2026-04-22 | kernel-fusion | attention, fp4, fp8 |
| [#13219](../sources/prs/TensorRT-LLM/PR-13219.md) | [TRTLLM-34871][feat] Add cute dsl FP8 paged MQA logits decode kernel | 2026-04-20 | persistent-kernel, pipeline-stages, swizzling | attention, fp8, gemm |
| [#13103](../sources/prs/TensorRT-LLM/PR-13103.md) | [None][feat] Optimize causal_conv1d prefill and decode kernels | 2026-04-16 |  |  |
| [#13117](../sources/prs/TensorRT-LLM/PR-13117.md) | [None][feat] Add FP4 residual quantization kernel without channel reo… | 2026-04-16 | swizzling | fp4, fp8, gemm |
| [#13033](../sources/prs/TensorRT-LLM/PR-13033.md) | [None][feat] Update rms_norm + fp4_qaunt kernel supporting more dim | 2026-04-14 | swizzling | fp4, moe, nvfp4 |
| [#13052](../sources/prs/TensorRT-LLM/PR-13052.md) | [#12716][feat] Fused cross-head QK Norm + RoPE kernel for WAN | 2026-04-14 | kernel-fusion | attention, reduction |
| [#12946](../sources/prs/TensorRT-LLM/PR-12946.md) | [#12784][feat] AutoDeploy: Optimize DeepSeek-R1 model performance | 2026-04-11 |  | attention, mla, moe |
| [#12937](../sources/prs/TensorRT-LLM/PR-12937.md) | [TRTLLM-11485][feat] Feature rework: Add SageAttention refreshed kernels (attentionOp only) | 2026-04-10 |  | attention, fp8, quantization |
| [#12884](../sources/prs/TensorRT-LLM/PR-12884.md) | [TRTLLM-11585][feat] Add CUTEDSL moe backend for nemotron-h | 2026-04-09 |  | block-scale, gemm, grouped-gemm |
| [#12731](../sources/prs/TensorRT-LLM/PR-12731.md) | [None][feat] Optimize mamba SSD prefill and extend flashinfer dispatch | 2026-04-03 |  | moe, topk |
| [#12642](../sources/prs/TensorRT-LLM/PR-12642.md) | [None][feat] Add triton paged attention for AutoDeploy | 2026-04-01 | pipeline-stages | attention, reduction |
| [#12612](../sources/prs/TensorRT-LLM/PR-12612.md) | [None][feat] Trtllm-gen FMHA JIT support | 2026-03-31 |  | attention, flash-attention, mla |
| [#12581](../sources/prs/TensorRT-LLM/PR-12581.md) | [https://nvbugs/5983390][perf] Multiple host perf optimizations for DSA part | 2026-03-30 |  | attention, fp8, sparse-attention |
| [#12537](../sources/prs/TensorRT-LLM/PR-12537.md) | [None][feat] Add Mamba2 MTP SSM cache CUDA kernel for tree-based speculative decoding | 2026-03-25 |  |  |
| [#12470](../sources/prs/TensorRT-LLM/PR-12470.md) | [None][feat] Support sparse mqa/gqa attention | 2026-03-24 |  | attention, flash-attention, mla |
| [#12506](../sources/prs/TensorRT-LLM/PR-12506.md) | [None][feat] Add PDL support to CuTE DSL top-k kernels | 2026-03-24 |  | gemm, topk |
| [#12456](../sources/prs/TensorRT-LLM/PR-12456.md) | [None][perf] add Dynamic SMEM block routing in MOE | 2026-03-23 |  | block-scale, moe, scan |
| [#12385](../sources/prs/TensorRT-LLM/PR-12385.md) | [None][feat] Temporally-Correlated Heuristic-guided Indexer TopK for Sparse Attention | 2026-03-20 |  | attention, fp4, nvfp4 |
| [#12354](../sources/prs/TensorRT-LLM/PR-12354.md) | [TRTLLM-10407][perf] Add cute dsl single pass multi cta cluster topk | 2026-03-19 |  | quantization, sort, topk |
| [#12320](../sources/prs/TensorRT-LLM/PR-12320.md) | [None][feat] Support update weight for nvfp4 | 2026-03-18 |  | attention, block-scale, fp4 |
| [#12322](../sources/prs/TensorRT-LLM/PR-12322.md) | [https://nvbugs/5983390][perf] Kernel fusions in _gather_k_cache_for_chunk of Indexer in DSA | 2026-03-18 | kernel-fusion | attention, fp4, fp8 |
| [#12236](../sources/prs/TensorRT-LLM/PR-12236.md) | [TRTLLM-10407][perf] Enable CuteDSL indexer_top_k in model | 2026-03-16 |  | reduction, sort, topk |
| [#12136](../sources/prs/TensorRT-LLM/PR-12136.md) | [None][feat] Add DWDP (Distributed Weight Data Parallelism) support for MoE inference | 2026-03-12 | double-buffering | block-scale, fp4, gemm |
| [#12062](../sources/prs/TensorRT-LLM/PR-12062.md) | [TRTLLM-11540][feat] Add EAGLE3 dynamic tree speculative decoding support | 2026-03-10 |  | attention, flash-attention, topk |
| [#12074](../sources/prs/TensorRT-LLM/PR-12074.md) | [TRTLLM-11289][feat] Integrate CuteDSL's bf16 dense GEMMs | 2026-03-10 | pipeline-stages, swizzling, tile-scheduling | attention, fp4, gemm |
| [#12079](../sources/prs/TensorRT-LLM/PR-12079.md) | [None][feat] CuteDSL MOE: Add raster along M/N support for blockscaled contiguous backbone kernel | 2026-03-10 | persistent-kernel, swizzling, tile-scheduling | block-scale, gemm, moe |
| [#11897](../sources/prs/TensorRT-LLM/PR-11897.md) | [TRTLLM-10990][feat] Fuse SwiGLU and quant into shared expert | 2026-03-04 | kernel-fusion | fp4, fp8, gemm |
| [#11899](../sources/prs/TensorRT-LLM/PR-11899.md) | [TRTLLM-10421][perf] Add fused cat+fp8_quantize CUDA kernel for DSA indexer | 2026-03-04 | kernel-fusion | fp8, quantization, reduction |
| [#11900](../sources/prs/TensorRT-LLM/PR-11900.md) | [TRTLLM-10407][feat] Integrate CuTE DSL top-k kernel for Blackwell | 2026-03-04 |  | scan, sort, topk |
| [#11869](../sources/prs/TensorRT-LLM/PR-11869.md) | [None][feat] Add fused DiT QK Norm + RoPE CUDA kernel for FLUX | 2026-03-03 | kernel-fusion | attention, fp4, fp8 |
| [#11733](../sources/prs/TensorRT-LLM/PR-11733.md) | [https://nvbugs/5799917][fix] Recover from CUTLASS MoE doActivation perf regression for MXFP4/NVFP4 dtype | 2026-02-26 | pipeline-stages | fp4, moe, nvfp4 |
| [#11697](../sources/prs/TensorRT-LLM/PR-11697.md) | [TRTLLM-11092][feat] add support for visual gen FA4 attention backend | 2026-02-25 | double-buffering, pipeline-stages, swizzling | attention, flash-attention, fp4 |
| [#11718](../sources/prs/TensorRT-LLM/PR-11718.md) | [TRTLLM-11119][feat] Blackwell SageAttention, Integrate into AttentionOp API | 2026-02-25 | pipeline-stages | attention, flash-attention, fp8 |
| [#11501](../sources/prs/TensorRT-LLM/PR-11501.md) | [None][feat] TRT-LLM Gen MoE finalize kernel optimization | 2026-02-13 | pipeline-stages | block-scale, moe |
| [#11510](../sources/prs/TensorRT-LLM/PR-11510.md) | [None][feat] Add support for expert_number<=2048 and K<=32 | 2026-02-13 | pipeline-stages | block-scale, moe, reduction |
| [#11473](../sources/prs/TensorRT-LLM/PR-11473.md) | [None][feat] Optimize by fuse nvfp4_quant to layernorm_gated for mamba2_mixer | 2026-02-12 | pipeline-stages, swizzling | fp4, nvfp4, quantization |
| [#11273](../sources/prs/TensorRT-LLM/PR-11273.md) | [None][feat] Optimize super-v3 nvfp4 for better perf | 2026-02-04 | kernel-fusion, pipeline-stages, swizzling | attention, flash-attention, fp4 |
| [#11181](../sources/prs/TensorRT-LLM/PR-11181.md) | [https://nvbugs/5854860][fix] Fix cutedsl argmax on sm120 | 2026-02-02 | pipeline-stages |  |
| [#11165](../sources/prs/TensorRT-LLM/PR-11165.md) | [https://nvbugs/5799917][fix] Recover from CUTLASS MoE doActivation perf regression for MXFP4/NVFP4 dtype | 2026-01-31 | pipeline-stages | fp4, fp8, moe |
| [#11143](../sources/prs/TensorRT-LLM/PR-11143.md) | [None][feat] fuse shared to sparse experts in TRT-LLM Gen MoE | 2026-01-30 | pipeline-stages | block-scale, moe, topk |
| [#10987](../sources/prs/TensorRT-LLM/PR-10987.md) | [TRTLLM-9831][perf] Use TMA.RED to improve effective memory bandwidth | 2026-01-26 | pipeline-stages, swizzling | block-scale, gemm, reduction |
| [#10742](../sources/prs/TensorRT-LLM/PR-10742.md) | [https://nvbugs/5669671][fix] Support GuidedDecoder with sharded logits (pick #10698) | 2026-01-16 | pipeline-stages |  |
| [#10476](../sources/prs/TensorRT-LLM/PR-10476.md) | [TRTLLM-10276][feat] Integrate cutedsl argmax kernel | 2026-01-07 |  | mbarrier, reduction |
| [#10479](../sources/prs/TensorRT-LLM/PR-10479.md) | [None] [feat] Add densegemm backend for MoE | 2026-01-07 | pipeline-stages, swizzling, tile-scheduling | block-scale, fp4, gemm |
| [#10429](../sources/prs/TensorRT-LLM/PR-10429.md) | [None] [feat] Add test script and raster M for gather fc1 kernel | 2026-01-06 | persistent-kernel, pipeline-stages, swizzling | block-scale |
| [#10327](../sources/prs/TensorRT-LLM/PR-10327.md) | [None][fix] impl fused triton kernel for e8m0 resmooth to reduce memory footprint | 2025-12-29 | pipeline-stages | fp8, quantization |
| [#10264](../sources/prs/TensorRT-LLM/PR-10264.md) | [TRTLLM-10022][feat] Add hopper xqa decode support for skip softmax attention | 2025-12-24 | pipeline-stages | attention |
| [#10190](../sources/prs/TensorRT-LLM/PR-10190.md) | [None][feat] sm100 weight-only kernel | 2025-12-22 | pipeline-stages, swizzling, warp-specialization | clc, fp8, gemm |
| [#10201](../sources/prs/TensorRT-LLM/PR-10201.md) | [TRTLLM-9831][perf] Enable 2CTA with autotune for CuteDSL MoE and Grouped GEMM optimizations | 2025-12-22 | double-buffering, pipeline-stages | block-scale, gemm, grouped-gemm |
| [#10130](../sources/prs/TensorRT-LLM/PR-10130.md) | [TRTLLM-9457][feat] Add cute dsl fp8 gemm for Blackwell | 2025-12-18 | pipeline-stages | fp4, fp8, gemm |
| [#10088](../sources/prs/TensorRT-LLM/PR-10088.md) | [None][feat] CuteDSL MOE FC1 Enhancement | 2025-12-17 | double-buffering, pipeline-stages, tile-scheduling | block-scale, gemm, moe |
| [#10042](../sources/prs/TensorRT-LLM/PR-10042.md) | [None][perf] Add more optimization options for MOE CuteDSL finalized kernel | 2025-12-16 | double-buffering, persistent-kernel, pipeline-stages | block-scale, grouped-gemm, moe |
| [#10043](../sources/prs/TensorRT-LLM/PR-10043.md) | [TRTLLM-9992][perf] Enable PDL for CuteDSL kernels and overlap MoeOutputMemset | 2025-12-16 | persistent-kernel, pipeline-stages | block-scale, moe |
| [#9905](../sources/prs/TensorRT-LLM/PR-9905.md) | [None][feat] Adding torch ext API for FusedAddRMSNormQuant kernel | 2025-12-11 | kernel-fusion, pipeline-stages | fp4, quantization |
| [#9924](../sources/prs/TensorRT-LLM/PR-9924.md) | [TRTLLM-9493][feat] Add helixPostProcessNative kernel for cp_dim=2 | 2025-12-11 | pipeline-stages |  |
| [#9838](../sources/prs/TensorRT-LLM/PR-9838.md) | [https://nvbugs/5726962][feat] Apply fusion for W4AFP8_AWQ MoE | 2025-12-09 | epilogue-fusion, pipeline-stages | block-scale, fp4, fp8 |
| [#9852](../sources/prs/TensorRT-LLM/PR-9852.md) | [None][feat] Fused kernels (qknormrope + moe routing) and two-model MTP support for glm4moe | 2025-12-09 | kernel-fusion, pipeline-stages | attention, fp4, moe |
| [#9854](../sources/prs/TensorRT-LLM/PR-9854.md) | [None][feat] Port fp4 quantization kernel optimization from FlashInfer | 2025-12-09 | pipeline-stages, swizzling | block-scale, fp4, quantization |
| [#9618](../sources/prs/TensorRT-LLM/PR-9618.md) | [TRTLLM-9685] [feat] Add gather fc1 kernel by cuteDSL | 2025-12-02 | pipeline-stages | block-scale, fp4, gemm |
| [#9175](../sources/prs/TensorRT-LLM/PR-9175.md) | [None][feat] TRT-LLM Gen MoE optimize DeepSeek Fp8 activation kernel | 2025-11-14 | pipeline-stages | block-scale, fp8, moe |
| [#9087](../sources/prs/TensorRT-LLM/PR-9087.md) | [None][fix] support topk autotuner input for expert slot per group larger than 32 | 2025-11-12 | pipeline-stages | block-scale, fp4, moe |
| [#8620](../sources/prs/TensorRT-LLM/PR-8620.md) | [None][feat] Enable nvfp4 cuda core for sm120 | 2025-10-23 | pipeline-stages | fp4, gemm, nvfp4 |
| [#8501](../sources/prs/TensorRT-LLM/PR-8501.md) | [None][fix] Fix the performance issue of FP8 blockwise grouped GEMM when using attention DP | 2025-10-20 | pipeline-stages | attention, block-scale, fp8 |
| [#8405](../sources/prs/TensorRT-LLM/PR-8405.md) | [TRTLLM-8535][feat] Support DeepSeek V3.2 with FP8 + BF16 KV cache/NVFP4 + BF16 KV cache | 2025-10-16 | pipeline-stages | attention, fp4, fp8 |
| [#7937](../sources/prs/TensorRT-LLM/PR-7937.md) | [None][feat] GPT-OSS Sm120/Sm121 Support | 2025-09-23 | pipeline-stages | attention, flash-attention, fp4 |
| [#7755](../sources/prs/TensorRT-LLM/PR-7755.md) | [None][fix] Fix and add test for TRTLLM MoE backend | 2025-09-16 | pipeline-stages | block-scale, fp8, moe |
| [#7761](../sources/prs/TensorRT-LLM/PR-7761.md) | [TRTLLM-8637][feat] Optimize the routing kernel for DeepseekV3 (MoE CUTLASS backend); Add support for 384 experts (MoE TRTLLM backend) | 2025-09-16 | pipeline-stages | block-scale, moe, reduction |
| [#7524](../sources/prs/TensorRT-LLM/PR-7524.md) | [None][chore] Fix kernel launch param and add TRTLLM MoE backend test | 2025-09-04 | pipeline-stages | block-scale, fp4, fp8 |
| [#6809](../sources/prs/TensorRT-LLM/PR-6809.md) | [OMNIML-2336][feat] Add NVFP4 x FP8 | 2025-08-12 | pipeline-stages, swizzling | fp4, fp8, gemm |
| [#4867](../sources/prs/TensorRT-LLM/PR-4867.md) | feat: Add w4a8_mxfp4_fp8 quantization recipe. | 2025-06-03 | persistent-kernel, pipeline-stages, tile-scheduling | fp4, gemm, quantization |

<a id="nvidiacccl"></a>
## NVIDIA/cccl
60 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#9056](../sources/prs/cccl/PR-9056.md) | Vectorize contiguous iterators in `cub::BlockLoad`/`Store` | 2026-05-18 |  |  |
| [#9019](../sources/prs/cccl/PR-9019.md) | [libcu++] Always suppress C++ extensions warnings in prologue | 2026-05-15 |  |  |
| [#8905](../sources/prs/cccl/PR-8905.md) | [STF] Add per-handle exec_place stream resources | 2026-05-12 |  | reduction |
| [#8925](../sources/prs/cccl/PR-8925.md) | Use the new tuning API internally for `detail::select|three_way_partition::dispatch` and `DevicePartition` | 2026-05-12 |  |  |
| [#8927](../sources/prs/cccl/PR-8927.md) | Use the new tuning API internally for `detail::segmented_radix_sort::dispatch` | 2026-05-12 |  |  |
| [#8880](../sources/prs/cccl/PR-8880.md) | Use the new tuning API internally for `detail::select::dispatch` and `DeviceSelect` | 2026-05-08 |  |  |
| [#8861](../sources/prs/cccl/PR-8861.md) | [cub] Simplify arch dispatch | 2026-05-07 |  |  |
| [#8839](../sources/prs/cccl/PR-8839.md) | Fix Warpspeed scan shifted output store | 2026-05-06 |  | scan |
| [#8826](../sources/prs/cccl/PR-8826.md) | Use the new tuning API internally for `detail::reduce[_nd]::dispatch[_nd]` | 2026-05-05 |  |  |
| [#8756](../sources/prs/cccl/PR-8756.md) | Use the new tuning API internally for `detail::reduce_by_key::dispatch` | 2026-04-30 |  |  |
| [#8742](../sources/prs/cccl/PR-8742.md) | Use the new tuning API internally for `detail::topk::dispatch` | 2026-04-29 |  | topk |
| [#8695](../sources/prs/cccl/PR-8695.md) | Replace `detail::segmented_reduce::dispatch` by the public API | 2026-04-27 |  |  |
| [#8565](../sources/prs/cccl/PR-8565.md) | Replace `detail::for_each::dispatch` by CUB's public API | 2026-04-21 |  |  |
| [#8538](../sources/prs/cccl/PR-8538.md) | Implement the new tuning API for `detail::batched_topk::dispatch_batched_topk` | 2026-04-20 |  |  |
| [#8495](../sources/prs/cccl/PR-8495.md) | Replace `detail::scan::dispatch` by CUB's public API | 2026-04-16 |  | scan |
| [#8473](../sources/prs/cccl/PR-8473.md) | Replace `detail::merge_sort::dispatch` by CUB's public API | 2026-04-15 |  | sort |
| [#8423](../sources/prs/cccl/PR-8423.md) | Vectorize mbarrier initialization in warpspeed scan | 2026-04-14 |  | mbarrier |
| [#8395](../sources/prs/cccl/PR-8395.md) | [CUB] Replace `Shuffle(Up|Down|Index)` with cuda::device::warp_shuffle - RadixSort only | 2026-04-13 |  |  |
| [#8355](../sources/prs/cccl/PR-8355.md) | [cub]: implement utilities for policy selection | 2026-04-10 |  | scan |
| [#8381](../sources/prs/cccl/PR-8381.md) | Replace `detail::merge::dispatch` by CUB's public API | 2026-04-10 |  |  |
| [#8352](../sources/prs/cccl/PR-8352.md) | Apply some random warpspeed tunings | 2026-04-09 |  |  |
| [#8332](../sources/prs/cccl/PR-8332.md) | simplify dispatch segmented reduce to use latest dispatch and new tunings API | 2026-04-08 |  | reduction |
| [#8311](../sources/prs/cccl/PR-8311.md) | Implement the new tuning API for `DispatchSelectIf` | 2026-04-07 |  | scan |
| [#8291](../sources/prs/cccl/PR-8291.md) | Port `thrust::min|max_element` to CUB | 2026-04-04 |  |  |
| [#8184](../sources/prs/cccl/PR-8184.md) | Avoid passing uninitialized values to scan_op | 2026-03-26 |  | reduction, scan, tma |
| [#8190](../sources/prs/cccl/PR-8190.md) | [STF] Move unstable_unique from STF to generic cudax utility | 2026-03-26 |  |  |
| [#8125](../sources/prs/cccl/PR-8125.md) | Optimized Device-to-Device Tensor Copy (cudax) - Transpose Case | 2026-03-20 |  | sort |
| [#8040](../sources/prs/cccl/PR-8040.md) | Adds support for non-fundamental types via decomposer to `DeviceTopK`  | 2026-03-16 |  | sort, topk |
| [#7928](../sources/prs/cccl/PR-7928.md) | Implement the new tuning API for `DispatchTopK` | 2026-03-09 | double-buffering | scan, topk |
| [#7940](../sources/prs/cccl/PR-7940.md) | [cuda.compute]: Fix faulty pointer arithmetic calculation in CUB dispatch | 2026-03-09 |  |  |
| [#7949](../sources/prs/cccl/PR-7949.md) | Use the new tuning API for `detail::radix_sort::dispatch` | 2026-03-09 | double-buffering | sort |
| [#7874](../sources/prs/cccl/PR-7874.md) | Implement the new tuning API for `DispatchSegmentedSort` | 2026-03-03 | double-buffering | scan, sort |
| [#7844](../sources/prs/cccl/PR-7844.md) | Implement the new tuning API for `DispatchSegmentedRadixSort` | 2026-03-02 | double-buffering | scan, sort |
| [#7823](../sources/prs/cccl/PR-7823.md) | Optimized Device-to-Device Tensor Copy (`cudax`) | 2026-02-27 |  | sort, tma |
| [#7805](../sources/prs/cccl/PR-7805.md) | Forward policy hub from `dispatch_streaming_arg_reduce_t` to `reduce::dispatch` | 2026-02-26 |  |  |
| [#7807](../sources/prs/cccl/PR-7807.md) | Implement the new tuning API for `detail::reduce::dispatch_streaming_arg_reduce_t` | 2026-02-26 | double-buffering | reduction |
| [#7810](../sources/prs/cccl/PR-7810.md) | Use the new tuning API internally for `detail::transform::dispatch` | 2026-02-26 |  |  |
| [#7814](../sources/prs/cccl/PR-7814.md) | [Backport branch/3.3.x] Forward policy hub from `dispatch_streaming_arg_reduce_t` to `reduce::dispatch` | 2026-02-26 |  |  |
| [#7795](../sources/prs/cccl/PR-7795.md) | Add env SegmentedReduce (non fixed-size overloads) | 2026-02-25 |  | reduction |
| [#7718](../sources/prs/cccl/PR-7718.md) | Optimize non fixed size segmented reduce for small segments using max_segment_size | 2026-02-19 |  |  |
| [#7669](../sources/prs/cccl/PR-7669.md) | Implement the new tuning API for `DeviceRleDispatch` | 2026-02-13 |  | scan |
| [#7384](../sources/prs/cccl/PR-7384.md) | Radix-selection based `BlockTopK` specialization | 2026-01-27 |  | scan, sort, topk |
| [#7346](../sources/prs/cccl/PR-7346.md) | Implement the new tuning API for deterministic (rfa) reduce dispatch | 2026-01-25 |  | reduction |
| [#7114](../sources/prs/cccl/PR-7114.md) | Two-phase reduction for fixed size segmented reduction for very large segment sizes | 2026-01-08 |  | reduction |
| [#7093](../sources/prs/cccl/PR-7093.md) | Implement new tuning API arch dispatching | 2026-01-06 |  |  |
| [#6819](../sources/prs/cccl/PR-6819.md) | Use integer promotion for `warp_reduce` | 2025-12-01 |  | reduction |
| [#6811](../sources/prs/cccl/PR-6811.md) | Integrate decoupled lookahead warpspeed scan | 2025-11-28 | warp-specialization | mbarrier, scan, tma |
| [#6597](../sources/prs/cccl/PR-6597.md) | Split fixed-size segmented reduce dispatch header | 2025-11-12 |  | reduction |
| [#6152](../sources/prs/cccl/PR-6152.md) | Fix debug section around line 390 of dispatch_topk | 2025-10-08 |  |  |
| [#6069](../sources/prs/cccl/PR-6069.md) | Add dynamic CUB dispatch for segmented_sort | 2025-09-30 |  | scan |
| [#6077](../sources/prs/cccl/PR-6077.md) | [CUB] Use `BlockLoadToShared` in `DeviceMerge` | 2025-09-30 |  | tma |
| [#5408](../sources/prs/cccl/PR-5408.md) | Combine `block_reduce_warp_reduction_nondeterministic.cuh` specialization with original deterministic one  | 2025-08-01 |  | reduction |
| [#5314](../sources/prs/cccl/PR-5314.md) | CUB - Add internal integer utils and tests (Split `WarpReduce` PR) | 2025-07-18 |  |  |
| [#4961](../sources/prs/cccl/PR-4961.md) | Add nondeterministic reduce that uses atomics | 2025-06-11 |  | reduction |
| [#4716](../sources/prs/cccl/PR-4716.md) | Split Optimize Warp Reduce PR - CUB part | 2025-05-15 |  |  |
| [#3691](../sources/prs/cccl/PR-3691.md) | Fix SM100 histogram tunings | 2025-02-05 |  |  |
| [#3559](../sources/prs/cccl/PR-3559.md) | Add b200 tunings for scan.exclusive.sum | 2025-01-28 |  |  |
| [#3517](../sources/prs/cccl/PR-3517.md) | Fix the vectorized loading of BlockLoad | 2025-01-24 |  | topk |
| [#3236](../sources/prs/cccl/PR-3236.md) | Fix scan / sm90 perf regression  | 2025-01-02 |  |  |
| [#2944](../sources/prs/cccl/PR-2944.md) | fix thread-reduce performance regression | 2024-11-22 |  | reduction |

<a id="nvidiacutlass"></a>
## NVIDIA/cutlass
36 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3176](../sources/prs/cutlass/PR-3176.md) | Small Tile N BlockScaled GEMM + Grouped GEMM on SM12x | 2026-04-19 | pipeline-stages | block-scale, gemm, grouped-gemm |
| [#3130](../sources/prs/cutlass/PR-3130.md) | Update blackwell tutorial to be compatible with 4.5-dev version | 2026-03-25 |  | gemm |
| [#3106](../sources/prs/cutlass/PR-3106.md) | [CLI] add cutedsl fp16 gemm tutorial from 2 to 6 | 2026-03-13 | pipeline-stages, swizzling, tile-scheduling | clc, gemm, quantization |
| [#3092](../sources/prs/cutlass/PR-3092.md) | Support for Group GEMM in CUTLASS Profiler for GeForce and Spark | 2026-03-07 | pipeline-stages | gemm |
| [#3091](../sources/prs/cutlass/PR-3091.md) | [Hopper CuTeDSL] Add grouped GEMM kernel example | 2026-03-06 | pipeline-stages, swizzling, tile-scheduling | fp8, gemm, grouped-gemm |
| [#3055](../sources/prs/cutlass/PR-3055.md) | Replace std::min with cute::min in sm120 blockwise scaling device functions | 2026-02-23 |  | gemm |
| [#3021](../sources/prs/cutlass/PR-3021.md) | [Cute-DSL] Add option for issue_clc_query without multicast | 2026-02-11 | swizzling | clc |
| [#2995](../sources/prs/cutlass/PR-2995.md) | [CuTeDSL] Fix: SM100 block-scale gemm overlapping accumulator | 2026-02-03 | pipeline-stages | block-scale, gemm, tmem |
| [#2965](../sources/prs/cutlass/PR-2965.md) | [Bug Fix]Set NumSplitsM to 1 when TileShapeM < 128 in sm90 fp8 blockwise scaling CollectiveMma | 2026-01-19 |  | fp8, gemm |
| [#2921](../sources/prs/cutlass/PR-2921.md) | Fix incorrect tensor layout strides in Blackwell MMA tutorial comments | 2026-01-03 |  |  |
| [#2881](../sources/prs/cutlass/PR-2881.md) | new example with TMA prefetch feature targeting for DRAM latency boun… | 2025-12-16 | pipeline-stages, swizzling, tile-scheduling | fp8, gemm, quantization |
| [#2875](../sources/prs/cutlass/PR-2875.md) | [cute] Add constexpr specifier to make_tiled_copy | 2025-12-12 |  |  |
| [#2865](../sources/prs/cutlass/PR-2865.md) | [Bug Fix]Bypass launch grids for SM120 Kernel with SM90 Mainloop & SM100 TileScheduler | 2025-12-09 | tile-scheduling | block-scale, gemm |
| [#2790](../sources/prs/cutlass/PR-2790.md) | Blockscaled Ragged Contiguous Grouped Gemm for MoEs | 2025-11-21 | tile-scheduling, warp-specialization | block-scale, fp4, gemm |
| [#2750](../sources/prs/cutlass/PR-2750.md) | Add tutorial fp16_gemm_1 | 2025-11-05 | pipeline-stages, swizzling | gemm, tcgen05, tma |
| [#2746](../sources/prs/cutlass/PR-2746.md) | Support for GEMM-K=0 for Blackwell Grouped GEMMs | 2025-11-04 |  | gemm |
| [#2719](../sources/prs/cutlass/PR-2719.md) | Support PDL for SM90 Array TMA GEMM | 2025-10-24 |  | gemm, grouped-gemm, tma |
| [#2713](../sources/prs/cutlass/PR-2713.md) | DistGEMM bug fixes | 2025-10-22 |  | gemm |
| [#2599](../sources/prs/cutlass/PR-2599.md) | fix gqa issue for blackwell fmha.py | 2025-08-28 |  | attention, flash-attention |
| [#2492](../sources/prs/cutlass/PR-2492.md) | fix: examples/cute/tutorial/blackwell/04_mma_tma_2sm_sm100.cu GridDim miscalculated | 2025-07-23 |  |  |
| [#2472](../sources/prs/cutlass/PR-2472.md) | Add Blackwell MLA forward (shape: d=192, dv=128) implementation | 2025-07-16 | persistent-kernel, tile-scheduling | attention, flash-attention, fp8 |
| [#2466](../sources/prs/cutlass/PR-2466.md) | Example 77 add blackwell fmha bwd for MLA shape | 2025-07-14 |  | attention, flash-attention, fp8 |
| [#2378](../sources/prs/cutlass/PR-2378.md) | support fp16 accmulator for sm89 fp8 mma | 2025-06-07 |  | fp8 |
| [#2366](../sources/prs/cutlass/PR-2366.md) | [ex77] fix mla split; add fwd lse; add bwd varlen | 2025-06-04 | persistent-kernel, tile-scheduling | attention, flash-attention, mla |
| [#2333](../sources/prs/cutlass/PR-2333.md) | Fix epilogue::thread::Convert cannot be used with DefaultEpilogue | 2025-05-26 |  | gemm |
| [#2270](../sources/prs/cutlass/PR-2270.md) | hopper-blockwise-generalization-optimization | 2025-04-29 | pipeline-stages, tile-scheduling, warp-specialization | block-scale, fp8, gemm |
| [#2220](../sources/prs/cutlass/PR-2220.md) | Set EpiTile correctly when TileN is not divisible by 32 | 2025-04-04 |  |  |
| [#2161](../sources/prs/cutlass/PR-2161.md) | Blockwise Improvement and Programmatic Dependent Launch | 2025-03-10 |  | gemm |
| [#2134](../sources/prs/cutlass/PR-2134.md) | Flash MLA Support - Step 2 | 2025-02-26 |  | mla |
| [#2139](../sources/prs/cutlass/PR-2139.md) | Blockwise and Groupwise GEMM for Blackwell and Improvements for Hopper | 2025-02-26 | swizzling, tile-scheduling, warp-specialization | block-scale, clc, fp8 |
| [#2130](../sources/prs/cutlass/PR-2130.md) | Flash MLA support | 2025-02-24 | double-buffering, swizzling, tile-scheduling | flash-attention, fp8, gemm |
| [#2123](../sources/prs/cutlass/PR-2123.md) | Hopper Grouped GEMM support for FP8 Accum | 2025-02-20 | pipeline-stages, warp-specialization | fp8, gemm, grouped-gemm |
| [#2095](../sources/prs/cutlass/PR-2095.md) | Improvements for: Groupwise scaling along M for FP8 gemm | 2025-02-10 |  | block-scale, fp8, gemm |
| [#2037](../sources/prs/cutlass/PR-2037.md) | Groupwise scaling along M for FP8 gemm | 2025-01-13 | swizzling, tile-scheduling, warp-specialization | block-scale, fp8, gemm |
| [#2033](../sources/prs/cutlass/PR-2033.md) | [EVT] Add support for Row/Col broadcast PtrArray | 2025-01-08 |  | fp8, gemm, grouped-gemm |
| [#1883](../sources/prs/cutlass/PR-1883.md) | Improve sm90 mixed dtype kernel | 2024-10-18 | warp-specialization | block-scale, fp8, gemm |

<a id="deepseek-aideepgemm"></a>
## deepseek-ai/DeepGEMM
8 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#328](../sources/prs/DeepGEMM/PR-328.md) | Sync nv_dev with upstream #316 (Mega MoE optimizations & benchmarks) | 2026-05-07 |  | attention, moe, tcgen05 |
| [#304](../sources/prs/DeepGEMM/PR-304.md) | [Public release 26/04] Introducing Mega MoE, FP4 Indexer and other features/fixes | 2026-04-16 | pipeline-stages, swizzling | block-scale, fp4, fp8 |
| [#193](../sources/prs/DeepGEMM/PR-193.md) | Fix multicast bug and optimize masked GEMM | 2025-09-12 |  | gemm |
| [#168](../sources/prs/DeepGEMM/PR-168.md) | Fix performance issue of m-grouped contiguous GEMMs. | 2025-08-20 |  | gemm |
| [#88](../sources/prs/DeepGEMM/PR-88.md) | Support TMA multicast on B with m_grouped_gemm_contiguous. | 2025-04-18 |  | gemm, tma |
| [#86](../sources/prs/DeepGEMM/PR-86.md) | Use swizzling instead of padding | 2025-04-14 | swizzling | tma, wgmma |
| [#83](../sources/prs/DeepGEMM/PR-83.md) | Use 1D TMA store instead of 3D | 2025-04-11 |  | tma, wgmma |
| [#78](../sources/prs/DeepGEMM/PR-78.md) |  Solving bank conflict via padding and TMA 3D store | 2025-04-03 |  | tma, wgmma |

<a id="flashinfer-aiflashinfer"></a>
## flashinfer-ai/flashinfer
322 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#3324](../sources/prs/flashinfer/PR-3324.md) | checkpointing_ssu kernel: fused replay + conditional state-write for Mamba2 | 2026-05-14 | double-buffering, pipeline-stages, swizzling | attention, block-scale, fp8 |
| [#3328](../sources/prs/flashinfer/PR-3328.md) | feat(cute_dsl/moe): add `moe_output_memset_inplace` dense memset wrapper | 2026-05-14 |  | gemm, moe |
| [#3286](../sources/prs/flashinfer/PR-3286.md) | feat(cute_dsl/moe): deterministic balanced autotune profile inputs | 2026-05-11 |  | fp4, fp8, moe |
| [#3276](../sources/prs/flashinfer/PR-3276.md) | fix(fmha_v2): fix FP8 V-scratch pipeline and varlen scheduler on SM90 | 2026-05-09 | persistent-kernel, pipeline-stages, warp-specialization | attention, flash-attention, fp8 |
| [#3268](../sources/prs/flashinfer/PR-3268.md) | Ameyn/gdn bf16 dispatcher and 4d pool | 2026-05-08 |  |  |
| [#3271](../sources/prs/flashinfer/PR-3271.md) | feat(moe): add SM120 W4A16 b12x kernels | 2026-05-08 | pipeline-stages, swizzling | block-scale, fp4, fp8 |
| [#3259](../sources/prs/flashinfer/PR-3259.md) | Add dynamic tokens-per-page TRTLLM-GEN GQA kernels | 2026-05-07 |  | attention, flash-attention, mla |
| [#3252](../sources/prs/flashinfer/PR-3252.md) | fix(cute_dsl/moe): unbias autotuner profiling for tile_size enumeration | 2026-05-06 |  | moe |
| [#3235](../sources/prs/flashinfer/PR-3235.md) | Support Kimi K2.5 H64 CuTe DSL MLA decode | 2026-05-05 |  | attention, mla, reduction |
| [#3237](../sources/prs/flashinfer/PR-3237.md) | perf: optimize per-token nvfp4 quantization kernel. | 2026-05-05 | swizzling | fp4, nvfp4, quantization |
| [#3216](../sources/prs/flashinfer/PR-3216.md) | fix(cute_dsl/moe): make autotuner bucket configuration adapt to runtime input | 2026-05-01 |  | gemm, moe |
| [#3191](../sources/prs/flashinfer/PR-3191.md) | fix(sm12x): fix micro-kernel workspace sizing when routed_rows > num_local_experts | 2026-04-27 |  | moe, topk |
| [#3181](../sources/prs/flashinfer/PR-3181.md) | cute-dsl fmha prefill (cubin integration): remove front-padding, add attention_sink, and pdl support | 2026-04-26 |  | attention, flash-attention, fp8 |
| [#3185](../sources/prs/flashinfer/PR-3185.md) | feat: enable glm5 router gemm | 2026-04-26 |  | gemm, moe |
| [#3151](../sources/prs/flashinfer/PR-3151.md) | perf: Add no-bias path for tinygemm_bf16 | 2026-04-23 |  | gemm |
| [#3152](../sources/prs/flashinfer/PR-3152.md) | Integrate CUTLASS Small Tile N Blockscaled GEMMs/Grouped GEMMs for SM120 and SM121 | 2026-04-23 | swizzling | block-scale, fp4, gemm |
| [#3157](../sources/prs/flashinfer/PR-3157.md) | feat: DiT layer norm fusions for WAN: flashinfer.diffusion_ops | 2026-04-23 | swizzling | attention, block-scale, fp4 |
| [#3129](../sources/prs/flashinfer/PR-3129.md) | feat: Enable FP8 (E4M3/E5M2) in concat_mla_k for optimize long-context prefill performance and refactor type dispatch for BF16/FP16 | 2026-04-21 |  | attention, flash-attention, fp4 |
| [#3097](../sources/prs/flashinfer/PR-3097.md) | Support NVFP4 KV for prefill and batch attention kernels | 2026-04-17 | swizzling | attention, block-scale, fp4 |
| [#3066](../sources/prs/flashinfer/PR-3066.md) | feat: Add b12x CuTe DSL fused MoE for SM120 | 2026-04-14 | kernel-fusion, persistent-kernel, pipeline-stages | block-scale, fp4, fp8 |
| [#3051](../sources/prs/flashinfer/PR-3051.md) | feat: Add backend="b12x" for mm_fp4 on SM120 | 2026-04-13 | pipeline-stages, swizzling, tile-scheduling | block-scale, fp4, gemm |
| [#3024](../sources/prs/flashinfer/PR-3024.md) | [feat] Add routing_replay_out support to MoE kernels and Python API | 2026-04-09 | kernel-fusion, pipeline-stages | block-scale, fp8, moe |
| [#3025](../sources/prs/flashinfer/PR-3025.md) | Prevent MoE autotuner buffer overflow on large token buckets | 2026-04-09 |  | fp4, fp8, gemm |
| [#3026](../sources/prs/flashinfer/PR-3026.md) | perf: Port TRT-LLM SM120/SM121 FP4 CUTLASS GEMM optimizations. Add PDL | 2026-04-09 | epilogue-fusion, persistent-kernel, pipeline-stages | block-scale, fp4, gemm |
| [#3027](../sources/prs/flashinfer/PR-3027.md) | [feat] Trtllm-gen Per-token Nvfp4 MoE | 2026-04-09 | swizzling | block-scale, fp4, fp8 |
| [#3014](../sources/prs/flashinfer/PR-3014.md) | perf: Optimize CUTLASS MoE helper kernels for small-batch decode workloads | 2026-04-08 |  | fp4, fp8, gemm |
| [#3001](../sources/prs/flashinfer/PR-3001.md) | [feat] Add blackwell GDN prefill kernel | 2026-04-07 | tile-scheduling | attention |
| [#3007](../sources/prs/flashinfer/PR-3007.md) | fix: use sym_int64 for strides in rmsnorm CuTe DSL kernels to prevent int32 overflow | 2026-04-07 |  | quantization |
| [#3008](../sources/prs/flashinfer/PR-3008.md) | feat: add PDL support to rmsnorm_fp4quant and add_rmsnorm_fp4quant CuTe DSL kernels | 2026-04-07 |  | fp4, quantization |
| [#2988](../sources/prs/flashinfer/PR-2988.md) | [Fmha] support nvfp4 output keepsMmaAb generation kernels | 2026-04-06 |  | attention, flash-attention, fp4 |
| [#2996](../sources/prs/flashinfer/PR-2996.md) | fix: tinygemm2 hang issue due to barrier sync | 2026-04-06 |  | mbarrier, reduction, tma |
| [#2962](../sources/prs/flashinfer/PR-2962.md) | Improved `simple` mamba SSU kernel  | 2026-04-02 | double-buffering, pipeline-stages | quantization, tma |
| [#2965](../sources/prs/flashinfer/PR-2965.md) | Add flashinfer.fused_rmsnorm_silu() with native kernel backend | 2026-04-02 |  | block-scale, fp4, fp8 |
| [#2944](../sources/prs/flashinfer/PR-2944.md) | feat: Add CuTe DSL grouped-gemm + combine fusion support | 2026-04-01 | persistent-kernel, pipeline-stages | block-scale, gemm, grouped-gemm |
| [#2945](../sources/prs/flashinfer/PR-2945.md) | fix: use float instead of double in sampling binary search to avoid FP64 bottleneck on SM103 | 2026-04-01 |  | topk |
| [#2914](../sources/prs/flashinfer/PR-2914.md) | feat: Add cuBLASLt backend for `mm_bf16` and enable multi-tactic autotuning for FP8/MXFP8 runners | 2026-03-30 |  | fp4, fp8, gemm |
| [#2913](../sources/prs/flashinfer/PR-2913.md) | [NVIDIA] fix(jit): enable GDC for CUTLASS fused MoE PDL — prevent random crashes on SM12x | 2026-03-29 |  | block-scale, fp4, fp8 |
| [#2908](../sources/prs/flashinfer/PR-2908.md) | feat(gdn): state checkpointing in chunk_gated_delta_rule | 2026-03-28 |  |  |
| [#2901](../sources/prs/flashinfer/PR-2901.md) | feat: add pdl support for cute dsl mla decode kernel support | 2026-03-27 |  | mla |
| [#2902](../sources/prs/flashinfer/PR-2902.md) | feat: add MXFP8 GEMM support for SM120 | 2026-03-27 | persistent-kernel, swizzling | block-scale, fp4, gemm |
| [#2904](../sources/prs/flashinfer/PR-2904.md) | perf: Optimize CuTe-DSL fp4 and fp8 quantization kernels | 2026-03-27 | pipeline-stages, swizzling | fp4, fp8, nvfp4 |
| [#2865](../sources/prs/flashinfer/PR-2865.md) | Mamba SSU: horizontal MTP kernel (+ DSTATE=96 support) | 2026-03-23 | double-buffering, pipeline-stages, swizzling | block-scale, mbarrier, tma |
| [#2836](../sources/prs/flashinfer/PR-2836.md) | [Fmha] Sparse MLA decode kernel selection heuristics | 2026-03-20 |  | attention, flash-attention, fp8 |
| [#2838](../sources/prs/flashinfer/PR-2838.md) | feat: Add CuTe-DSL backend for NVFP4 quantization | 2026-03-20 | pipeline-stages, swizzling, warp-specialization | fp4, fp8, gemm |
| [#2811](../sources/prs/flashinfer/PR-2811.md) | CuteDSL MoE fix redundant output buffer zeroing | 2026-03-18 | pipeline-stages | block-scale, fp4, moe |
| [#2805](../sources/prs/flashinfer/PR-2805.md) | [CuTe DSL] Add modular FMHA prefill and MLA decode attention kernels | 2026-03-17 | double-buffering, persistent-kernel, pipeline-stages | attention, flash-attention, fp8 |
| [#2792](../sources/prs/flashinfer/PR-2792.md) | feat: Support padding tokens with seqlen=0 for rope+quant+kv cache update fusion kernel | 2026-03-16 |  | attention, mla, quantization |
| [#2777](../sources/prs/flashinfer/PR-2777.md) | perf: Performance tune cute dsl RMSNorm variants | 2026-03-13 |  | fp8, mbarrier, quantization |
| [#2779](../sources/prs/flashinfer/PR-2779.md) | feat: FP8 output support for CUTLASS MLA paged attention | 2026-03-13 |  | attention, fp8, gemm |
| [#2752](../sources/prs/flashinfer/PR-2752.md) | [feat] Add air top-p algorithm | 2026-03-11 |  | sort, topk |
| [#2757](../sources/prs/flashinfer/PR-2757.md) | feat: Add FP4 KV cache quant/dequant kernels  | 2026-03-11 | swizzling | block-scale, fp4, fp8 |
| [#2738](../sources/prs/flashinfer/PR-2738.md) | Support for MXFP4 and NVFP4 group GEMMs on GeForce and Spark | 2026-03-10 | swizzling | block-scale, fp4, fp8 |
| [#2743](../sources/prs/flashinfer/PR-2743.md) | Add cute dsl mla decode op | 2026-03-10 | persistent-kernel, tile-scheduling | attention, fp8, mla |
| [#2744](../sources/prs/flashinfer/PR-2744.md) | [feat] Add 2048 experts and 32 Top K  | 2026-03-10 |  | moe, reduction, scan |
| [#2727](../sources/prs/flashinfer/PR-2727.md) | [gdn] support non-contiguous state for decoding | 2026-03-09 |  |  |
| [#2702](../sources/prs/flashinfer/PR-2702.md) | Add NVFP4 KV cache quantization support for SM100 | 2026-03-06 |  | attention, flash-attention, fp4 |
| [#2709](../sources/prs/flashinfer/PR-2709.md) | Mamba2 SSD Combined Forward Pass (Blackwell CuTe DSL Kernel) | 2026-03-06 | kernel-fusion, persistent-kernel, tile-scheduling | attention, scan, tma |
| [#2711](../sources/prs/flashinfer/PR-2711.md) | feat: Add DiT-oriented kernels where Qk (Bmm1) type can be reinterpreted into Int8 or BFloat16 | 2026-03-06 |  | attention, flash-attention |
| [#2700](../sources/prs/flashinfer/PR-2700.md) | Add varlen and speculative decoding support to selective state update | 2026-03-05 |  | quantization |
| [#2670](../sources/prs/flashinfer/PR-2670.md) | fix: reduce smem allocation for tinygemm2 kernel in SM120 | 2026-03-03 | pipeline-stages |  |
| [#2661](../sources/prs/flashinfer/PR-2661.md) | feat: implement deterministic topk | 2026-03-01 |  | scan, sort, topk |
| [#2653](../sources/prs/flashinfer/PR-2653.md) | [feat] trtllm-gen mxfp8 gemm | 2026-02-28 | swizzling | fp4, fp8, gemm |
| [#2660](../sources/prs/flashinfer/PR-2660.md) | feat: support mxfp4 & mxfp8 entrypoint for blackwell cutedsl dense gemm | 2026-02-28 | pipeline-stages, swizzling | block-scale, fp4, fp8 |
| [#2642](../sources/prs/flashinfer/PR-2642.md) | [fp8_blockwise]Fix int32 overflow in TRTLLM fused MoE activation kernel | 2026-02-26 | pipeline-stages | moe, topk |
| [#2645](../sources/prs/flashinfer/PR-2645.md) | int16 Block-Scaled State and Stochastic Rounding for SSU (mamba) | 2026-02-26 |  | block-scale, quantization, reduction |
| [#2629](../sources/prs/flashinfer/PR-2629.md) | fix: cute dsl nvfp4 moe routing index error | 2026-02-24 |  | fp4, moe, nvfp4 |
| [#2631](../sources/prs/flashinfer/PR-2631.md) | fix: add SM121 support to SM120 version guards | 2026-02-24 |  | gemm |
| [#2618](../sources/prs/flashinfer/PR-2618.md) | perf(gdn): optimize MTP kernel with ILP rows and SMEM v caching | 2026-02-22 | pipeline-stages | reduction, tma |
| [#2610](../sources/prs/flashinfer/PR-2610.md) | Ameyn/gdn bf16 tolerance parallel reduction | 2026-02-21 |  | reduction |
| [#2605](../sources/prs/flashinfer/PR-2605.md) | [bugfix] Fix FilteredTopK overflow correctness | 2026-02-20 |  | sort, topk |
| [#2587](../sources/prs/flashinfer/PR-2587.md) | feat: trtllm tinygemm2 in flashinfer as bf16 routergemm | 2026-02-19 |  | gemm, mbarrier |
| [#2591](../sources/prs/flashinfer/PR-2591.md) | Mamba SSU: better automatic kernel selection + algorithm selection optionally exposed to the user. | 2026-02-19 |  | tma |
| [#2581](../sources/prs/flashinfer/PR-2581.md) | Implement `cutlass_fused_moe` mxfp8 | 2026-02-18 | swizzling | block-scale, fp4, fp8 |
| [#2573](../sources/prs/flashinfer/PR-2573.md) | [Bug] Fix spark unit test failures for test_add_rmsnorm_fp4_quant_cute_dsl | 2026-02-17 |  |  |
| [#2564](../sources/prs/flashinfer/PR-2564.md) | fix: W4A8 autotune crash in cutlass_fused_moe profiler workspace | 2026-02-14 |  | fp4, fp8, moe |
| [#2540](../sources/prs/flashinfer/PR-2540.md) | feat: cute dsl mmfp4 for blackwell | 2026-02-11 | pipeline-stages, swizzling, tile-scheduling | block-scale, fp4, gemm |
| [#2520](../sources/prs/flashinfer/PR-2520.md) | Support NVFP4 KV cache decode on SM120 | 2026-02-08 | swizzling | fp4, nvfp4, quantization |
| [#2521](../sources/prs/flashinfer/PR-2521.md) | Feat/gdn decode pooled | 2026-02-08 |  | attention, reduction |
| [#2509](../sources/prs/flashinfer/PR-2509.md) | perf: cache cudaGetDeviceProperties in gdn_prefill to avoid per-call overhead | 2026-02-06 |  |  |
| [#2498](../sources/prs/flashinfer/PR-2498.md) | Ameyn/gdn decode cutedsl kernel | 2026-02-05 |  | attention, reduction |
| [#2503](../sources/prs/flashinfer/PR-2503.md) | refactor: Port upstream CUTLASS fixes and refactor grouped_gemm_nt_masked GEMM module location | 2026-02-05 |  | block-scale, gemm |
| [#2477](../sources/prs/flashinfer/PR-2477.md) | feat: Add TRTLLM-Gen Skip-Softmax kernels for prefill and decode | 2026-02-03 |  | attention, flash-attention |
| [#2460](../sources/prs/flashinfer/PR-2460.md) | perf: add fp4 GEMM tile configs and streamK scheduler for SM120 | 2026-02-02 | stream-k, tile-scheduling | fp4, gemm |
| [#2462](../sources/prs/flashinfer/PR-2462.md) | feat: Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron, fixed | 2026-02-02 |  | fp4, fp8, moe |
| [#2464](../sources/prs/flashinfer/PR-2464.md) | feat: Add MXFP8 GEMM mm_mxfp8 (cutlass) | 2026-02-02 | persistent-kernel, swizzling | block-scale, fp4, fp8 |
| [#2456](../sources/prs/flashinfer/PR-2456.md) | fix: fix illegal memory access for NaN input in sampling kernels | 2026-01-31 |  |  |
| [#2443](../sources/prs/flashinfer/PR-2443.md) | Add cute-dsl backends to mxfp[8,4]_quantization for future refactor | 2026-01-30 | swizzling | fp4, fp8, nvfp4 |
| [#2444](../sources/prs/flashinfer/PR-2444.md) | MTP for mamba  | 2026-01-30 | pipeline-stages, swizzling | gemm, tma |
| [#2446](../sources/prs/flashinfer/PR-2446.md) | feat: Add TRTLLM fmha_v2 library for SM90 attention with Skip-Softmax  | 2026-01-30 | kernel-fusion, swizzling | attention, flash-attention, fp8 |
| [#2432](../sources/prs/flashinfer/PR-2432.md) | fix: Sampling: CUDA Graph fix | 2026-01-29 |  |  |
| [#2441](../sources/prs/flashinfer/PR-2441.md) | fix: Fix NaN output in mxfp8_quantize for very small input values | 2026-01-29 |  | quantization |
| [#2428](../sources/prs/flashinfer/PR-2428.md) | refactor: refactoring cuda code to cute-dsl (part 1) | 2026-01-28 |  | fp8, quantization |
| [#2421](../sources/prs/flashinfer/PR-2421.md) | refactor: simplify fp4 rmsnorm | 2026-01-27 |  | fp4, fp8, mbarrier |
| [#2422](../sources/prs/flashinfer/PR-2422.md) | refactor: reduce hopper's gdn prefill compilation time and fix docstring. | 2026-01-27 |  |  |
| [#2415](../sources/prs/flashinfer/PR-2415.md) | Remove cudaMalloc/Free in GDN prefill kernel | 2026-01-25 |  | tma |
| [#2405](../sources/prs/flashinfer/PR-2405.md) | perf: improve gdn decode cute-dsl kernels | 2026-01-23 | pipeline-stages | reduction |
| [#2398](../sources/prs/flashinfer/PR-2398.md) | feat: cuteDSL fp4 moe for better DSR1 performance. | 2026-01-22 |  | fp4, fp8, moe |
| [#2395](../sources/prs/flashinfer/PR-2395.md) | feat: Add output_both_sf_layouts option to add_rmsnorm_fp4quant API | 2026-01-21 | swizzling | block-scale, fp4, nvfp4 |
| [#2378](../sources/prs/flashinfer/PR-2378.md) | bugfix: hotfix of PR 2366 (mamba kernel) | 2026-01-20 |  |  |
| [#2380](../sources/prs/flashinfer/PR-2380.md) | fix: ensure each CTA processes full numHeadsQPerKv for trtllm decode kernel | 2026-01-20 |  | attention, flash-attention |
| [#2385](../sources/prs/flashinfer/PR-2385.md) | fix: In-place Residual Update for add_rmsnorm_fp4quant | 2026-01-20 |  | fp4, quantization |
| [#2387](../sources/prs/flashinfer/PR-2387.md) | A Blackwell-optimized version of selective_state_update (mamba) | 2026-01-20 | pipeline-stages | reduction, tma |
| [#2370](../sources/prs/flashinfer/PR-2370.md) | feat: [Qwen3-Next] Add Cute DSL GDN decode kernel and  tests | 2026-01-18 | pipeline-stages, swizzling | attention, reduction, tma |
| [#2366](../sources/prs/flashinfer/PR-2366.md) | Enable fp16/bf16/f32 support for selective_state_update (mamba) | 2026-01-16 |  | tma |
| [#2343](../sources/prs/flashinfer/PR-2343.md) | Optimize quantization function in large problem size | 2026-01-13 | swizzling, warp-specialization | block-scale, fp4, fp8 |
| [#2327](../sources/prs/flashinfer/PR-2327.md) | [perf] Improve gemm_fp8_nt_groupwise (cutlass backend) by 10-40% for batch sizes <= 32 | 2026-01-11 |  | fp8, gemm, tcgen05 |
| [#2328](../sources/prs/flashinfer/PR-2328.md) | fix: guard batchWarpReduceSum with ENABLE_FP8 to fix compilation without FP8 | 2026-01-11 |  | fp8 |
| [#2325](../sources/prs/flashinfer/PR-2325.md) | bugfix: fix multi-cta top-k implementation when k value is different for different row | 2026-01-10 |  | sort, topk |
| [#2323](../sources/prs/flashinfer/PR-2323.md) | [ML3] Optimized Router Gemm | 2026-01-09 |  | gemm, moe |
| [#2308](../sources/prs/flashinfer/PR-2308.md) | Fix: FilteredTopKUnifiedKernel read value out of length | 2026-01-08 |  | topk |
| [#2303](../sources/prs/flashinfer/PR-2303.md) | [Perf][Feature] Add SM103-specific schedulers for NVFP4 CUTLASS kernels | 2026-01-07 | persistent-kernel | fp4, gemm, nvfp4 |
| [#2304](../sources/prs/flashinfer/PR-2304.md) | feat: Support Fused MoE non gated Relu2 NVFP4 & FP8 and support Nemotron | 2026-01-07 |  | fp4, fp8, gemm |
| [#2301](../sources/prs/flashinfer/PR-2301.md) | Selective State Update kernel (mamba) | 2026-01-06 |  | tma |
| [#2279](../sources/prs/flashinfer/PR-2279.md) | [WIP] Refactor: simplify torch -> cute-dsl boilerplate and enable tvm-ffi for cute-dsl kernels | 2026-01-01 | swizzling | fp4 |
| [#2276](../sources/prs/flashinfer/PR-2276.md) | feat: add GDN Attention | 2025-12-31 |  | attention, gemm |
| [#2265](../sources/prs/flashinfer/PR-2265.md) | [TRTLLM-Gen Fmha] add optimized trtllm-gen decode kernels for high throughput + speculative decoding | 2025-12-24 |  | attention, flash-attention, mla |
| [#2260](../sources/prs/flashinfer/PR-2260.md) | fix: Add global scale support and optional output allocation for RMSNorm+FP4Quant fusion kernels | 2025-12-23 | swizzling | block-scale, fp4, nvfp4 |
| [#2255](../sources/prs/flashinfer/PR-2255.md) | fix: support int64 IdType for RoPE part argument in `rope_quantize_fp8_append_paged_kv_cache` | 2025-12-22 |  |  |
| [#2243](../sources/prs/flashinfer/PR-2243.md) | feat: RMSNorm/Fused RMSNorm + FP8 Quantization kernels | 2025-12-19 |  | fp8, quantization |
| [#2244](../sources/prs/flashinfer/PR-2244.md) | Remove cudaStreamSynchronize from gemm_groupwise_sm120.cuh for CUDA graph compatibility | 2025-12-19 |  | gemm |
| [#2237](../sources/prs/flashinfer/PR-2237.md) | [feat] Integrate SGLang concat_mla_k kernel into flashinfer | 2025-12-18 |  | attention, mla |
| [#2233](../sources/prs/flashinfer/PR-2233.md) | feat: Fused RMSNorm + FP4 Quantization Kernels in CuTe-DSL | 2025-12-17 | kernel-fusion, swizzling | block-scale, fp4, fp8 |
| [#2215](../sources/prs/flashinfer/PR-2215.md) | feat: further optimize top-k and add fused top-k page construction kernels for DSA | 2025-12-13 |  | sort, topk |
| [#2159](../sources/prs/flashinfer/PR-2159.md) | feat: MxInt4 x Bf16 TRT-LLM Gen MoE support | 2025-12-02 |  | block-scale, gemm, moe |
| [#2149](../sources/prs/flashinfer/PR-2149.md) | enable sm103 moe dsl backend | 2025-11-28 |  | block-scale, moe |
| [#2142](../sources/prs/flashinfer/PR-2142.md) | feat: TRTLLM FMHAv2 backend for ctx attention | 2025-11-25 |  | attention, flash-attention, fp8 |
| [#2138](../sources/prs/flashinfer/PR-2138.md) | feat: add trtllm-gen per-tensor sparseMla kernels. | 2025-11-24 |  | attention, flash-attention, mla |
| [#2131](../sources/prs/flashinfer/PR-2131.md) | make DeepGEMM swapAB available for linear gemm SM90 | 2025-11-22 |  | block-scale, fp8, gemm |
| [#2126](../sources/prs/flashinfer/PR-2126.md) | fix flaky xqa test | 2025-11-21 |  | attention |
| [#2119](../sources/prs/flashinfer/PR-2119.md) | perf: bunch of features and optimizations for top-k (sampling + sparse attention) | 2025-11-20 | double-buffering | attention, reduction, sort |
| [#2109](../sources/prs/flashinfer/PR-2109.md) | feat: support more head dim in RoPE kernel | 2025-11-19 |  |  |
| [#2110](../sources/prs/flashinfer/PR-2110.md) | add tensor scale input for xqa | 2025-11-19 |  | attention, fp8, quantization |
| [#2111](../sources/prs/flashinfer/PR-2111.md) | refactor: update fa3 codebase and fix hopper unittest [part 1] | 2025-11-19 | double-buffering | attention, fp8, quantization |
| [#2114](../sources/prs/flashinfer/PR-2114.md) | feature: make the LSE returned by MLA support base 2 or e  #2113 | 2025-11-19 |  | attention, mla |
| [#2105](../sources/prs/flashinfer/PR-2105.md) | enable xqa speculative decoding | 2025-11-18 |  | attention |
| [#2090](../sources/prs/flashinfer/PR-2090.md) | refactor: pass hopper deepgemm include directory through python | 2025-11-14 |  | moe |
| [#2092](../sources/prs/flashinfer/PR-2092.md) | perf: TRT-LLM Gen finalize kernel optimization | 2025-11-14 |  | moe, topk |
| [#2079](../sources/prs/flashinfer/PR-2079.md) | [Feature] Support batch prefill for POD Attention | 2025-11-12 |  | attention |
| [#2081](../sources/prs/flashinfer/PR-2081.md) | enable xqa fp8 output | 2025-11-12 |  | fp8 |
| [#2070](../sources/prs/flashinfer/PR-2070.md) | feat: BF16 GEMM using CUTLASS backend for SM100 | 2025-11-10 |  | gemm |
| [#2058](../sources/prs/flashinfer/PR-2058.md) | perf: Optimize helper max/minmax function in sampling.cuh | 2025-11-07 |  | reduction |
| [#2062](../sources/prs/flashinfer/PR-2062.md) | Fix: several bugs/issues with trtllm-gen attention kernels.  | 2025-11-07 |  | attention, flash-attention, mla |
| [#2063](../sources/prs/flashinfer/PR-2063.md) | perf: TRT-LLM MoE Block-FP8 activation optimization | 2025-11-07 |  | block-scale, fp8, moe |
| [#2051](../sources/prs/flashinfer/PR-2051.md) | Add support for topkPacked input in block-level renormalize | 2025-11-06 |  |  |
| [#2053](../sources/prs/flashinfer/PR-2053.md) | feat: add xqa mla backend | 2025-11-06 |  | fp8, mla |
| [#2044](../sources/prs/flashinfer/PR-2044.md) | perf: improve sampling/mask/softmax performance (part 1/2) | 2025-11-05 |  | reduction, sort, topk |
| [#2047](../sources/prs/flashinfer/PR-2047.md) | Rebase FP8 SM100 Cutlass FMHA Attention to main (original PR#1238) | 2025-11-05 |  | attention, flash-attention, fp8 |
| [#2049](../sources/prs/flashinfer/PR-2049.md) | [BUG] Fix trtllm-gen fp4 moe renormalize routing | 2025-11-05 |  | fp4, moe, quantization |
| [#2033](../sources/prs/flashinfer/PR-2033.md) | use scalar for kv_scale in xqa | 2025-11-04 |  | fp8 |
| [#2037](../sources/prs/flashinfer/PR-2037.md) | feat: Add flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache (fused RoPE + Q + KV cache, supports MLA/GQA/MHA)  | 2025-11-04 |  | attention, fp8, mla |
| [#2025](../sources/prs/flashinfer/PR-2025.md) | perf: Speed up fp4 quantization for small batch with swizzling for cutlass MoE | 2025-11-03 | swizzling | block-scale, fp4, moe |
| [#2028](../sources/prs/flashinfer/PR-2028.md) | [NVIDIA] Thor & Spark Support | 2025-11-03 |  |  |
| [#2019](../sources/prs/flashinfer/PR-2019.md) | [DSV3] Optimized Router Gemm | 2025-10-31 |  | gemm, reduction |
| [#2020](../sources/prs/flashinfer/PR-2020.md) | update trtllm cutlass moe  | 2025-10-31 | epilogue-fusion, swizzling | block-scale, fp4, fp8 |
| [#2014](../sources/prs/flashinfer/PR-2014.md) | [feat] Refactor trtllmgen MOE and add Bf16 trtllmgen moe | 2025-10-30 |  | fp4, fp8, moe |
| [#2001](../sources/prs/flashinfer/PR-2001.md) | feat: add xqa backend and completes NHD/HND coverage for trtllm-gen/xqa backend | 2025-10-29 |  | attention, fp8 |
| [#1994](../sources/prs/flashinfer/PR-1994.md) | minor fix for xqa | 2025-10-28 |  | mla |
| [#1982](../sources/prs/flashinfer/PR-1982.md) | fix: correct PDL parameter handling in RopeQuantize kernel | 2025-10-26 |  | quantization |
| [#1969](../sources/prs/flashinfer/PR-1969.md) | feat: enable deepgemm jit for fp8 block-scale on SM90 | 2025-10-23 |  | block-scale, fp8 |
| [#1973](../sources/prs/flashinfer/PR-1973.md) | Feature: Add support for L40 FusedMoE in cutlass path | 2025-10-23 |  | fp4, fp8, gemm |
| [#1954](../sources/prs/flashinfer/PR-1954.md) | Feature: Support Relu2 activation in fused MoE | 2025-10-20 |  | fp8, gemm, moe |
| [#1955](../sources/prs/flashinfer/PR-1955.md) | Update trtllm-gen fused moe routing kernel and add more kernels | 2025-10-20 |  | block-scale, fp4, fp8 |
| [#1926](../sources/prs/flashinfer/PR-1926.md) | Add layernorm op for inputs of mixed dtype | 2025-10-14 |  | fp8, quantization, topk |
| [#1927](../sources/prs/flashinfer/PR-1927.md) | silu_and_mul nvfp4 quanization fusion rework | 2025-10-14 | swizzling | block-scale, fp4, gemm |
| [#1924](../sources/prs/flashinfer/PR-1924.md) | MLA RoPE + quantization fused kernel: shape generalization for MHA / GQA | 2025-10-13 | kernel-fusion | attention, fp8, mla |
| [#1882](../sources/prs/flashinfer/PR-1882.md) | feat: Add FP4 TRTLLM-Gen throughput MOE batched gemms | 2025-10-07 |  | fp4, gemm, moe |
| [#1878](../sources/prs/flashinfer/PR-1878.md) | Tune kernel compilation parameters for https://github.com/flashinfer-ai/flashinfer/pull/1850  | 2025-10-06 | pipeline-stages | attention |
| [#1865](../sources/prs/flashinfer/PR-1865.md) | Bugfix: fix o_strides in persistent kernel  | 2025-10-04 | persistent-kernel | attention |
| [#1850](../sources/prs/flashinfer/PR-1850.md) | Add head_dim=64 for blackwell cutlass fmha implementation | 2025-10-03 |  | attention, flash-attention |
| [#1826](../sources/prs/flashinfer/PR-1826.md) | Bugfix: Fix data hazard in persistent reduce | 2025-10-01 |  | attention |
| [#1829](../sources/prs/flashinfer/PR-1829.md) | feat: trtrllm-gen global scaled FP8 GEMMs | 2025-10-01 |  | fp8, gemm |
| [#1831](../sources/prs/flashinfer/PR-1831.md) | Update the routing for TRTLLMGEN to support kimi k2 and qwen | 2025-10-01 |  | moe, reduction, scan |
| [#1835](../sources/prs/flashinfer/PR-1835.md) | [Quantization] Add per-expert global scaling factor for fp4 batched quantize | 2025-10-01 |  | fp4, quantization |
| [#1819](../sources/prs/flashinfer/PR-1819.md) | feat:enable fp8 blockscale moe for fused cultass for sm90 | 2025-09-30 | pipeline-stages, swizzling | block-scale, fp8, gemm |
| [#1769](../sources/prs/flashinfer/PR-1769.md) | feat: add xqa fp8 mha and fp8 kv cache | 2025-09-25 | swizzling | attention, fp8, gemm |
| [#1774](../sources/prs/flashinfer/PR-1774.md) | Masked batch nvfp4 quantization | 2025-09-25 |  | block-scale, fp4, nvfp4 |
| [#1723](../sources/prs/flashinfer/PR-1723.md) | Fix DeepSeek quality for TRTLLM fused MoE routing | 2025-09-19 |  | moe |
| [#1724](../sources/prs/flashinfer/PR-1724.md) | bugfix: partially fix tests/test_trtllm_gen_fused_moe.py unit test failure | 2025-09-19 |  | topk |
| [#1725](../sources/prs/flashinfer/PR-1725.md) | TVM: support TVM binding for GroupedGemm | 2025-09-19 |  | fp8, gemm, grouped-gemm |
| [#1727](../sources/prs/flashinfer/PR-1727.md) | fix: put sampling kernel launch into macro | 2025-09-19 |  |  |
| [#1685](../sources/prs/flashinfer/PR-1685.md) | perf: Port the separate reduce kernel mode from trtllm. | 2025-09-16 |  | attention, flash-attention, fp8 |
| [#1696](../sources/prs/flashinfer/PR-1696.md) | Support Kimi-K2 for TRT: templatize number of experts | 2025-09-16 |  | gemm, grouped-gemm, sort |
| [#1682](../sources/prs/flashinfer/PR-1682.md) | Update TGV GEMM default kernel and TGV code cleanup. | 2025-09-15 |  | gemm |
| [#1677](../sources/prs/flashinfer/PR-1677.md) | Support output signals for overlapping for cutedsl gemm | 2025-09-14 |  | block-scale, gemm |
| [#1670](../sources/prs/flashinfer/PR-1670.md) | feat: Add `variant.OutputTransform()` to decode kernels | 2025-09-11 | epilogue-fusion | attention, fp8 |
| [#1675](../sources/prs/flashinfer/PR-1675.md) | feat: Batch-size invariant FA2 Prefill & Decode | 2025-09-11 |  | attention |
| [#1668](../sources/prs/flashinfer/PR-1668.md) | TGV GEMM as a BF16 backend alternative to cuBLAS | 2025-09-10 | swizzling, tile-scheduling | attention, fp8, gemm |
| [#1661](../sources/prs/flashinfer/PR-1661.md) | perf&bugfix: skip kv-tile computation out of sliding window in FA2; fix __syncthreads in mergestate | 2025-09-09 |  | attention |
| [#1631](../sources/prs/flashinfer/PR-1631.md) | bugfix: trtllm-gen fmha sm101 and sm100 compatibility | 2025-09-03 |  | attention, flash-attention |
| [#1622](../sources/prs/flashinfer/PR-1622.md) | bugfix: collect all modules to aot | 2025-09-02 |  |  |
| [#1614](../sources/prs/flashinfer/PR-1614.md) | bugfix: fix merge_attention_state in BatchAttention w/ gqa-group-size in Qwen family | 2025-09-01 |  | attention |
| [#1608](../sources/prs/flashinfer/PR-1608.md) | feat: initial support for SM103, SM110, SM120, SM121 | 2025-08-30 |  | attention, flash-attention |
| [#1609](../sources/prs/flashinfer/PR-1609.md) | feat: cutlass fp4 gemm bringup for SM120 & SM121 | 2025-08-30 | persistent-kernel | block-scale, fp4, gemm |
| [#1610](../sources/prs/flashinfer/PR-1610.md) | feat: cutlass fp8 gemm bringup for SM120 & SM121 | 2025-08-30 |  | fp8, gemm, grouped-gemm |
| [#1611](../sources/prs/flashinfer/PR-1611.md) | bugfix: fix fp4 quantization with 8x4 scale factor layout | 2025-08-30 | swizzling | fp4, quantization |
| [#1596](../sources/prs/flashinfer/PR-1596.md) | bugfix: fix fused-temperature softmax IMA issue | 2025-08-28 |  |  |
| [#1597](../sources/prs/flashinfer/PR-1597.md) | bugfix: fix the register overflow issue for topk renorm kernels on blackwell | 2025-08-28 |  | topk |
| [#1582](../sources/prs/flashinfer/PR-1582.md) | bugfix: Fix arg passing to TORCH_CHECK and TORCH_WARN macros | 2025-08-26 |  | topk |
| [#1567](../sources/prs/flashinfer/PR-1567.md) | Backend: downgrade trtllm-gen kernel to cuda-12 | 2025-08-25 |  | attention, flash-attention |
| [#1571](../sources/prs/flashinfer/PR-1571.md) | bugfix: fix cuda version guard macros | 2025-08-25 |  |  |
| [#1573](../sources/prs/flashinfer/PR-1573.md) | update trtllm-gen fp4 autotuner and routing | 2025-08-25 |  | fp4 |
| [#1559](../sources/prs/flashinfer/PR-1559.md) | bugfix: fix persistent attention kernel correctness on blackwell | 2025-08-24 |  | attention |
| [#1565](../sources/prs/flashinfer/PR-1565.md) | fix: separate out fp4 lib into sm90 and sm100 versions, add oob checking in fused moe | 2025-08-24 |  | fp4, moe, quantization |
| [#1547](../sources/prs/flashinfer/PR-1547.md) | perf: replace cudaGetDeviceProperties with cudaDeviceGetAttribute | 2025-08-22 |  |  |
| [#1530](../sources/prs/flashinfer/PR-1530.md) | bugfix: Fix compile error for undefined swizzle enum. | 2025-08-21 | swizzling | quantization |
| [#1533](../sources/prs/flashinfer/PR-1533.md) | bugfix: Fix Persistent kernel precision for masked output  | 2025-08-21 | persistent-kernel | attention |
| [#1537](../sources/prs/flashinfer/PR-1537.md) | feat: Integrate TRTLLM varlen kernel for deepseek R1 prefill  | 2025-08-21 |  | attention, flash-attention |
| [#1518](../sources/prs/flashinfer/PR-1518.md) | backend: Refactor trtllm-gen fmha metainfo loading | 2025-08-20 |  | attention, flash-attention |
| [#1521](../sources/prs/flashinfer/PR-1521.md) | refactor fp4 masked gemm cute-dsl implementation and add manual cache | 2025-08-20 |  | block-scale, fp4, fp8 |
| [#1525](../sources/prs/flashinfer/PR-1525.md) | Add GeGLU support to trtllm-gen NVFP4 Fused MoE Kernel | 2025-08-20 |  | fp4, moe, nvfp4 |
| [#1503](../sources/prs/flashinfer/PR-1503.md) | feat: integrate xqa attention backend | 2025-08-18 | swizzling | attention, fp8, gemm |
| [#1509](../sources/prs/flashinfer/PR-1509.md) | bugfix: Fix stream handling in cutedsl gemm | 2025-08-18 |  | block-scale, gemm |
| [#1500](../sources/prs/flashinfer/PR-1500.md) | fix: Replace cub Max/Min with cuda::maximum/minimum for cuda 13 compatibility | 2025-08-16 |  | reduction |
| [#1498](../sources/prs/flashinfer/PR-1498.md) | feat: scaling at fp4 gemm epilogue | 2025-08-15 |  | block-scale, fp4, gemm |
| [#1483](../sources/prs/flashinfer/PR-1483.md) | perf: add fast path to TopPRenormProbKernel for top_p >= 1.0, significantly boosting SGLang workloads | 2025-08-14 |  |  |
| [#1488](../sources/prs/flashinfer/PR-1488.md) | fix: update cutedsl masked moe gemm | 2025-08-14 |  | block-scale, fp4, fp8 |
| [#1490](../sources/prs/flashinfer/PR-1490.md) | feat: Support fp8 qkv, fp16/bf16 out MHA for trtllm-gen. | 2025-08-14 |  | attention, flash-attention, fp8 |
| [#1481](../sources/prs/flashinfer/PR-1481.md) | Add python API for masked grouped gemm | 2025-08-13 |  | block-scale, gemm, grouped-gemm |
| [#1475](../sources/prs/flashinfer/PR-1475.md) | tuner: Trtllm-gen Fp4 MoE Autotunner | 2025-08-12 |  | fp4, fp8, moe |
| [#1445](../sources/prs/flashinfer/PR-1445.md) | Add alignment in MxFP8Quantization | 2025-08-10 | swizzling | block-scale, fp4, quantization |
| [#1446](../sources/prs/flashinfer/PR-1446.md) | Remove getEnvEnablePDL in favor of enable_pdl parameter | 2025-08-10 |  | attention, block-scale, flash-attention |
| [#1396](../sources/prs/flashinfer/PR-1396.md) | gpt-oss: Add MXFP8 x MXFP4 CUTLASS MOE for SM100 and BF16 x MXFP4 CUTLASS for SM90 + SwigluBias Activation | 2025-08-06 | swizzling | block-scale, fp4, fp8 |
| [#1397](../sources/prs/flashinfer/PR-1397.md) | feature: add cutlass as bmm_fp8 backend. | 2025-08-06 |  | fp8, gemm |
| [#1402](../sources/prs/flashinfer/PR-1402.md) | fix shared memory alignment conflict in sampling.cuh | 2025-08-06 |  |  |
| [#1389](../sources/prs/flashinfer/PR-1389.md) | GPT-OSS Support: Add Blackwell MoE mxfp4 implementation from TRTLLM and Attention Sink | 2025-08-05 | swizzling | attention, block-scale, fp4 |
| [#1376](../sources/prs/flashinfer/PR-1376.md) | bugfix: Add guard for fp4/fp8 related include headers | 2025-08-04 |  | fp4, fp8 |
| [#1371](../sources/prs/flashinfer/PR-1371.md) | bugfix: fixed cutlass fused moe usage of FP4QuantizationSFLayout::SWIZZLED | 2025-08-03 | swizzling | moe |
| [#1360](../sources/prs/flashinfer/PR-1360.md) | support trtllm-gen prefill fp4 output | 2025-07-31 |  | attention, flash-attention, fp4 |
| [#1355](../sources/prs/flashinfer/PR-1355.md) | feature: add fp4 mm using trtllm backend | 2025-07-30 | swizzling | fp4, quantization |
| [#1339](../sources/prs/flashinfer/PR-1339.md) | feat: Fused rope fp8 quantize kernel for MLA | 2025-07-28 |  | fp8, mla, quantization |
| [#1324](../sources/prs/flashinfer/PR-1324.md) | feat: Support logits_soft_cap for Persistent attn; fix kv split limit | 2025-07-25 |  | attention |
| [#1331](../sources/prs/flashinfer/PR-1331.md) | feat: masked layout fp4 gemm using cute-dsl | 2025-07-25 | tile-scheduling | block-scale, fp4, gemm |
| [#1320](../sources/prs/flashinfer/PR-1320.md) | Add blockwise-scaled FP8 GEMM via TRTLLM-Gen. | 2025-07-24 | swizzling | block-scale, fp4, fp8 |
| [#1322](../sources/prs/flashinfer/PR-1322.md) | feat: Add k_scale and v_scale to persistent attention  | 2025-07-24 |  | attention |
| [#1307](../sources/prs/flashinfer/PR-1307.md) | Fix the bug of the kernel-selection heuristic in trtllm-gen | 2025-07-23 |  | attention, flash-attention, mla |
| [#1305](../sources/prs/flashinfer/PR-1305.md) | [Feature] SM level profiler  | 2025-07-22 |  |  |
| [#1294](../sources/prs/flashinfer/PR-1294.md) | Update cutlass fp4 moe kernels | 2025-07-21 | swizzling | fp4, fp8, gemm |
| [#1296](../sources/prs/flashinfer/PR-1296.md) | add cutlass backend for mm_fp4 | 2025-07-21 | persistent-kernel, warp-specialization | fp4, gemm, grouped-gemm |
| [#1290](../sources/prs/flashinfer/PR-1290.md) | [fix] fix integer overflow in FA2 customized_mask & add buffer overflow warning. | 2025-07-19 |  | attention, quantization |
| [#1292](../sources/prs/flashinfer/PR-1292.md) | refactor: Improved metainfo for trtllm-gen fmha | 2025-07-19 |  | attention, flash-attention |
| [#1287](../sources/prs/flashinfer/PR-1287.md) | Bug fix: guard fp8 e8m0 and e2m1 compile  | 2025-07-18 |  | fp8, quantization |
| [#1289](../sources/prs/flashinfer/PR-1289.md) | refactor: refactor trtllm-gen attention kernel integration code | 2025-07-18 |  | attention, flash-attention, mla |
| [#1267](../sources/prs/flashinfer/PR-1267.md) | Bug fix: fix duplicate launch in POD | 2025-07-16 |  | attention |
| [#1251](../sources/prs/flashinfer/PR-1251.md) | Reduce the JIT compilation time of gen_gemm_sm100_module | 2025-07-14 |  | gemm |
| [#1240](../sources/prs/flashinfer/PR-1240.md) | Patch fp8 cubin availability | 2025-07-11 |  | fp8 |
| [#1241](../sources/prs/flashinfer/PR-1241.md) | feat: Support MXFP8 x MXFP4 CUTLASS grouped GEMM | 2025-07-11 | swizzling | fp4, gemm, grouped-gemm |
| [#1242](../sources/prs/flashinfer/PR-1242.md) | Add trtllm-gen attention mha kernel with FP8 Q/K/V and FP8 output | 2025-07-11 |  | attention, flash-attention, fp4 |
| [#1230](../sources/prs/flashinfer/PR-1230.md) | feat: Add non-causal cudnn prefill kernels | 2025-07-08 |  |  |
| [#1234](../sources/prs/flashinfer/PR-1234.md) | bugfix: support uint8_t for vec_t class template | 2025-07-08 |  | fp8 |
| [#1221](../sources/prs/flashinfer/PR-1221.md) | Enable cudnn decode and add tests for the cudnn decode kernel | 2025-07-07 |  | tma |
| [#1222](../sources/prs/flashinfer/PR-1222.md) | feat: add trtllm-gen mla cubin | 2025-07-07 |  | attention, flash-attention, mla |
| [#1227](../sources/prs/flashinfer/PR-1227.md) | Fix missing hash in the cudnn cubin path | 2025-07-07 |  | attention, flash-attention |
| [#1214](../sources/prs/flashinfer/PR-1214.md) | Feature/sm100 low latency nvfp4 kernels | 2025-07-04 | swizzling | block-scale, fp4, fp8 |
| [#1212](../sources/prs/flashinfer/PR-1212.md) | feat: trtllm-gen fp8 moe kernels | 2025-07-03 |  | block-scale, fp8, moe |
| [#1208](../sources/prs/flashinfer/PR-1208.md) | Fix the issue with auxillary kernel launch and grid dim calculation | 2025-07-02 |  |  |
| [#1206](../sources/prs/flashinfer/PR-1206.md) | [fix] fix BatchAttention CTA_TILE_KV mask issue | 2025-07-01 |  | attention |
| [#1198](../sources/prs/flashinfer/PR-1198.md) | bugfix: fix blackwell fmha hanging issue for empty kv_len | 2025-06-30 |  | attention, flash-attention |
| [#1200](../sources/prs/flashinfer/PR-1200.md) | [feat] optimize persistent batch attention perf. | 2025-06-30 |  | attention, reduction |
| [#1189](../sources/prs/flashinfer/PR-1189.md) | update trtllm-gen decode attention kernel launcher | 2025-06-28 |  | attention, flash-attention, mla |
| [#1178](../sources/prs/flashinfer/PR-1178.md) | bugfix: softmax NaN results caused by large -inf masks | 2025-06-25 |  | topk |
| [#1158](../sources/prs/flashinfer/PR-1158.md) | Add more logging to TRTLLM-GEN debug trace (NFC) | 2025-06-19 | tile-scheduling | attention, flash-attention |
| [#1153](../sources/prs/flashinfer/PR-1153.md) | feat: Fused temperature online softmax kernel | 2025-06-18 |  | reduction |
| [#1140](../sources/prs/flashinfer/PR-1140.md) | Fix FA2 and FA3 multi-item scoring and cuda illegal memory access error | 2025-06-12 |  | attention |
| [#1137](../sources/prs/flashinfer/PR-1137.md) | [feat] add unified batch attention w/ correctness tests. | 2025-06-11 | persistent-kernel, pipeline-stages, swizzling | attention, reduction |
| [#1116](../sources/prs/flashinfer/PR-1116.md) | hotfix: fix the blackwell fmha stream | 2025-06-06 |  | attention, flash-attention |
| [#1117](../sources/prs/flashinfer/PR-1117.md) | [Feature] Support PDL for batch Prefill and Decode | 2025-06-06 |  | attention, mla, quantization |
| [#1114](../sources/prs/flashinfer/PR-1114.md) | bugfix: Fix test and output shape of fp4 quantize | 2025-06-05 |  | fp4, gemm, quantization |
| [#1113](../sources/prs/flashinfer/PR-1113.md) | Add CUTLASS fused moe kernels from TensorRT-LLM. | 2025-06-04 | epilogue-fusion, pipeline-stages, swizzling | fp4, fp8, gemm |
| [#1106](../sources/prs/flashinfer/PR-1106.md) | bugfix: host-precomuted plan function for blackwell fmha | 2025-05-31 |  | attention, flash-attention, scan |
| [#1086](../sources/prs/flashinfer/PR-1086.md) | perf: accelerate blackwell grouped gemm | 2025-05-23 |  | gemm, grouped-gemm, moe |
| [#1087](../sources/prs/flashinfer/PR-1087.md) | bugfix: fix fp8 attention kernels aot compilation issue | 2025-05-23 |  | attention, fp8, quantization |
| [#1071](../sources/prs/flashinfer/PR-1071.md) | bugfix: adding lse output to blackwell fmha kernels | 2025-05-20 |  | attention, flash-attention |
| [#1072](../sources/prs/flashinfer/PR-1072.md) | bugfix: follow user-specified sm_scale for blackwell cutlass fmha | 2025-05-20 |  | attention, flash-attention |
| [#1059](../sources/prs/flashinfer/PR-1059.md) | Parameterize prefix mask call (needed by POD-Attention) | 2025-05-14 |  | attention |
| [#1054](../sources/prs/flashinfer/PR-1054.md) | Fix KV chunking for POD.  | 2025-05-13 |  | attention |
| [#1055](../sources/prs/flashinfer/PR-1055.md) | bugfix: temporally disable split-kv in blackwell mla | 2025-05-13 |  | attention, mla |
| [#1050](../sources/prs/flashinfer/PR-1050.md) | fix: top_k_mask_logits hangs on -inf inputs | 2025-05-09 |  |  |
| [#1051](../sources/prs/flashinfer/PR-1051.md) | [nvidia] Add Blackwell FMHA decode kernel from TRT-LLM | 2025-05-09 | persistent-kernel, tile-scheduling | attention, flash-attention, mla |
| [#1039](../sources/prs/flashinfer/PR-1039.md) | [nvidia] initial support for blackwell kernels | 2025-04-24 |  | attention, flash-attention, tma |
| [#1033](../sources/prs/flashinfer/PR-1033.md) | feat: add functional per-head FP8 quantization for FA3 | 2025-04-23 | pipeline-stages, tile-scheduling | attention, fp8, gemm |
| [#1035](../sources/prs/flashinfer/PR-1035.md) | feat: Softmax free sampling | 2025-04-23 |  | scan |
| [#1029](../sources/prs/flashinfer/PR-1029.md) | fix: add zero init for KV tiled copy | 2025-04-21 |  | attention |
| [#1025](../sources/prs/flashinfer/PR-1025.md) | feat: ragged tensor padding kernel for blackwell kernel alignment | 2025-04-20 |  | scan |
| [#1015](../sources/prs/flashinfer/PR-1015.md) | add multi-item scoring | 2025-04-11 | tile-scheduling | attention, quantization |
| [#1014](../sources/prs/flashinfer/PR-1014.md) | misc: fix instrument code for mla profiler | 2025-04-10 |  | attention, mla |
| [#1007](../sources/prs/flashinfer/PR-1007.md) | feat: update decode attention APIs | 2025-04-07 |  | attention |
| [#997](../sources/prs/flashinfer/PR-997.md) | 3rdparty: upgrade cutlass to 3.9 | 2025-04-03 |  | attention |
| [#991](../sources/prs/flashinfer/PR-991.md) | perf: prefetch page indices for mla kernel | 2025-03-31 |  | attention, mla |
| [#982](../sources/prs/flashinfer/PR-982.md) | SM-constraint-GEMM by triton persistent kernel | 2025-03-29 | persistent-kernel | gemm |
| [#983](../sources/prs/flashinfer/PR-983.md) | Triton `rms_norm` kernels | 2025-03-29 |  |  |
| [#974](../sources/prs/flashinfer/PR-974.md) | perf: dual pivot top-p/top-k renorm | 2025-03-26 |  | topk |
| [#969](../sources/prs/flashinfer/PR-969.md) | perf: Fix python API overhead when CUDAGraph is not enabled | 2025-03-23 |  |  |
| [#952](../sources/prs/flashinfer/PR-952.md) | perf: Use 2WG pipeline design for MLA implementation on Hopper | 2025-03-17 | pipeline-stages, swizzling | attention, mla, wgmma |
| [#945](../sources/prs/flashinfer/PR-945.md) | bugfix: fix potential issues of FA3 template loading nans for PageAttention | 2025-03-14 |  | attention |
| [#930](../sources/prs/flashinfer/PR-930.md) | feat: experimenta support of PDL | 2025-03-11 |  |  |
| [#913](../sources/prs/flashinfer/PR-913.md) | feat: flashinfer intra-kernel profiler | 2025-03-05 |  | attention, mla |
| [#901](../sources/prs/flashinfer/PR-901.md) | perf: tweak the pipeline design of mla kernel | 2025-02-27 | pipeline-stages | attention, mla |
| [#898](../sources/prs/flashinfer/PR-898.md) | perf: fix MLA split-k performance bug | 2025-02-25 |  | attention, mla |
| [#887](../sources/prs/flashinfer/PR-887.md) | perf: FlashAttention-3 style MLA PageAttention | 2025-02-23 | pipeline-stages, swizzling, warp-specialization | attention, flash-attention, mla |
| [#888](../sources/prs/flashinfer/PR-888.md) | feat - support mla kvcache store | 2025-02-23 |  | mla |
| [#869](../sources/prs/flashinfer/PR-869.md) | Naive Support for Hopper FP8 Prefill Kernel with Per-Head Quantization | 2025-02-18 | pipeline-stages, tile-scheduling | attention, flash-attention, fp8 |
| [#858](../sources/prs/flashinfer/PR-858.md) | Add POD-Attention to FlashInfer | 2025-02-17 | swizzling | attention |
| [#863](../sources/prs/flashinfer/PR-863.md) | perf: dynamic split-k for MLA | 2025-02-17 | swizzling | attention, mla |
| [#868](../sources/prs/flashinfer/PR-868.md) | bugfix: fix the behavior of MLA kernel when kv-length is 0 | 2025-02-17 |  | attention, mla |
| [#844](../sources/prs/flashinfer/PR-844.md) | perf: MLA decode kernel implemented by CuTe targeted to SM80 | 2025-02-14 | pipeline-stages, swizzling | attention, gemm, mla |
| [#821](../sources/prs/flashinfer/PR-821.md) | bugfix: bugfix on sm89 MLA | 2025-02-13 |  | attention, mla |
| [#810](../sources/prs/flashinfer/PR-810.md) | bugfix: mla page-attention kernel for different page sizes | 2025-02-12 |  | attention, mla |
| [#812](../sources/prs/flashinfer/PR-812.md) | feat: unlocking MLA for A100 | 2025-02-12 | swizzling | attention, mla |
| [#814](../sources/prs/flashinfer/PR-814.md) | feat: unlock MLA attention for sm89 (L40/L40s/4090) | 2025-02-12 | swizzling | attention, mla |
| [#804](../sources/prs/flashinfer/PR-804.md) | perf: memory efficient deepseek mla fused page-attention kernel | 2025-02-10 | pipeline-stages, swizzling | attention, mla |
| [#801](../sources/prs/flashinfer/PR-801.md) | feat: apply sm_scale at logits instead of q in FA2 template | 2025-02-09 |  | attention |
| [#799](../sources/prs/flashinfer/PR-799.md) | feat: support f32 attention output in FA2 template | 2025-02-08 |  | attention |
| [#793](../sources/prs/flashinfer/PR-793.md) | fix rope logic in mla decoding | 2025-02-07 |  | attention, mla |
| [#787](../sources/prs/flashinfer/PR-787.md) | bugfix: MLA decode should multiply sm_scale by math::log2e | 2025-02-05 |  | attention, mla |
| [#785](../sources/prs/flashinfer/PR-785.md) | bugfix: drop CTA_TILE_Q=32 | 2025-02-04 |  |  |
| [#776](../sources/prs/flashinfer/PR-776.md) | perf: refactor fa2 prefill template | 2025-02-03 |  | attention |
| [#774](../sources/prs/flashinfer/PR-774.md) | bugfix: Ensure Loop Termination by Enforcing IEEE-754 Compliance in Sampling Kernels | 2025-02-01 |  |  |
| [#765](../sources/prs/flashinfer/PR-765.md) | feat: support deepseek prefill attention shape | 2025-01-30 |  | attention, gemm, mla |
| [#754](../sources/prs/flashinfer/PR-754.md) | Change `apply_rope_with_cos_sin_cache` to accept `cos_sin_cache` | 2025-01-27 |  |  |
| [#728](../sources/prs/flashinfer/PR-728.md) | Align KV chunk size binary search with actual KV chunk splitting. | 2025-01-09 |  | attention |
| [#718](../sources/prs/flashinfer/PR-718.md) | bugfix: FusedAddRMSNorm kernels might require more than 48KB shared memory when d is large. | 2025-01-06 |  |  |
| [#714](../sources/prs/flashinfer/PR-714.md) | perf: fix the iteration bound of SWA in FA2 prefill template | 2025-01-03 |  | attention |

<a id="pytorchpytorch"></a>
## pytorch/pytorch
4 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#152967](../sources/prs/pytorch/PR-152967.md) | [ATen][CUDA] Optimize 128 bit vectorization | 2025-05-06 |  |  |
| [#150676](../sources/prs/pytorch/PR-150676.md) | [CUDA][avgpool2d] Fix backward launch bounds again for `sm100`, `sm120` | 2025-04-04 |  |  |
| [#150705](../sources/prs/pytorch/PR-150705.md) | [CUDA] Only use vec128 if CUDA version is newer than 12.8 | 2025-04-04 |  |  |
| [#150640](../sources/prs/pytorch/PR-150640.md) | [CUDA][avgpool2d] Fix backward launch bounds again for `sm100`, `sm120` | 2025-04-03 |  |  |

<a id="sgl-projectsglang"></a>
## sgl-project/sglang
169 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#25821](../sources/prs/sglang/PR-25821.md) | [Refactor] Rename NSA → DSA: user-facing aliases, file/class/import rename | 2026-05-19 |  | attention, sparse-attention |
| [#25554](../sources/prs/sglang/PR-25554.md) | amd/deepseek_v4 27/N [fix] Reduce Triton autotune configs for faster first-time server launch | 2026-05-18 | kernel-fusion | attention, fp8, mla |
| [#25695](../sources/prs/sglang/PR-25695.md) | fix (jit kernel): elementwise activation C++ error | 2026-05-18 |  |  |
| [#25532](../sources/prs/sglang/PR-25532.md) | [fp8] SM90 swap-AB scaled_mm dispatch (~1.16x kernel geomean, +5.8-18.5% end-to-end) | 2026-05-17 | persistent-kernel, stream-k | attention, fp8, gemm |
| [#25336](../sources/prs/sglang/PR-25336.md) | [Intel GPU] Enable DeepSeek V4 Inference on XPU | 2026-05-15 | pipeline-stages | attention, block-scale, fp4 |
| [#24933](../sources/prs/sglang/PR-24933.md) | Amd/deepseek v4 rebase main 0509 | 2026-05-11 | pipeline-stages, tile-scheduling | attention, flash-attention, fp8 |
| [#24986](../sources/prs/sglang/PR-24986.md) | [rebase]Deepseek_v4 support w4(mxfp4)a16 on hopper | 2026-05-11 |  | attention, fp4, gemm |
| [#24696](../sources/prs/sglang/PR-24696.md) | [Gemma4] Optimize Gemm4 with fused Q/K/V RMSNorm + per-expert FP8 ckpt loader | 2026-05-08 | kernel-fusion | attention, fp8, moe |
| [#24710](../sources/prs/sglang/PR-24710.md) | [codex] Optimize hidden-size 512 RMSNorm dispatch | 2026-05-08 |  | reduction |
| [#24490](../sources/prs/sglang/PR-24490.md) | Port MXFP4 Marlin MoE support to JIT kernel path | 2026-05-06 |  | fp4, fp8, gemm |
| [#24411](../sources/prs/sglang/PR-24411.md) | [diffusion] Fuse LTX2 split rotary embedding | 2026-05-05 | kernel-fusion |  |
| [#24271](../sources/prs/sglang/PR-24271.md) | [KDA] Optimize prefill kernels with diagonal and recompute fuse | 2026-05-02 |  | attention, scan |
| [#24048](../sources/prs/sglang/PR-24048.md) | [VLM] Optimize Gemma4 VLM with PCG and fuse RMSNorm + residual add + scalar | 2026-04-29 | pipeline-stages | moe |
| [#23938](../sources/prs/sglang/PR-23938.md) | Optimize large GroupNorm SiLU apply | 2026-04-28 |  |  |
| [#23961](../sources/prs/sglang/PR-23961.md) | [feat] Init true on policy with qwen_dense | 2026-04-28 |  | attention, moe, reduction |
| [#23965](../sources/prs/sglang/PR-23965.md) | Enable PDL for various kernels in DSV32/GLM5 | 2026-04-28 |  | gemm |
| [#23686](../sources/prs/sglang/PR-23686.md) | Deepseek_v4 support w4(mxfp4)a16 on hopper | 2026-04-25 |  | fp4, fp8, moe |
| [#22931](../sources/prs/sglang/PR-22931.md) | [Fix/Kernel] Add JIT rmsnorm_hf kernel to fix transformers backend MMLU accuracy regression  | 2026-04-16 |  | quantization, reduction |
| [#22814](../sources/prs/sglang/PR-22814.md) | diffusion: add HunyuanVideo GroupNorm+SiLU fast path | 2026-04-14 |  |  |
| [#22064](../sources/prs/sglang/PR-22064.md) | [Diffusion] Fix weight scale swizzle and add large-M kernel config for FLUX.2-dev-NVFP4 | 2026-04-03 | swizzling | fp4, gemm, nvfp4 |
| [#22079](../sources/prs/sglang/PR-22079.md) | [nvidia] Gemma4 nvfp4 fix | 2026-04-03 |  | attention, fp4, fp8 |
| [#21834](../sources/prs/sglang/PR-21834.md) | [Feature] JIT rmsnorm update (with claude) | 2026-04-01 |  |  |
| [#21654](../sources/prs/sglang/PR-21654.md) | Fused_qknorm_rope kernel optimization: up to 2.4× faster | 2026-03-30 | pipeline-stages | attention, reduction |
| [#21668](../sources/prs/sglang/PR-21668.md) | [XPU] Enable qwen3.5 on XPU | 2026-03-30 |  | attention, sort |
| [#21511](../sources/prs/sglang/PR-21511.md) | [AMD] Enable FP8 KV cache and FP8 attention kernel for NSA on MI300/MI355 with TileLang backend | 2026-03-27 |  | attention, fp8, gemm |
| [#21440](../sources/prs/sglang/PR-21440.md) | [Diffusion] Add qknorm rope fuse kernel | 2026-03-26 | kernel-fusion | attention, reduction |
| [#21411](../sources/prs/sglang/PR-21411.md) | [GDN] Fuse GDN kkt + solve_tril into one kernel | 2026-03-25 | kernel-fusion | attention, sort, topk |
| [#21314](../sources/prs/sglang/PR-21314.md) | CUTLASS NVFP4 GEMM improvement of SM120 | 2026-03-24 | stream-k | fp4, fp8, gemm |
| [#21203](../sources/prs/sglang/PR-21203.md) | [KDA] Support CuTeDSL KDA decode kernel | 2026-03-23 |  |  |
| [#21019](../sources/prs/sglang/PR-21019.md) | [Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel | 2026-03-20 | kernel-fusion |  |
| [#20887](../sources/prs/sglang/PR-20887.md) | CUTLASS FP8 Blockwise GEMM improvement of SM120 | 2026-03-18 |  | fp8, gemm, topk |
| [#20661](../sources/prs/sglang/PR-20661.md) | Fix(jit): support rmsnorm for hidden_size in {64, 128, 256} | 2026-03-16 |  |  |
| [#20673](../sources/prs/sglang/PR-20673.md) | [Feature][JIT Kernel] Fused TP QK norm For Minimax | 2026-03-16 |  |  |
| [#20479](../sources/prs/sglang/PR-20479.md) | Support Triton MLA FP8 KV cache | 2026-03-13 |  | attention, fp4, fp8 |
| [#20501](../sources/prs/sglang/PR-20501.md) | [Kernel] Fuse temperature + softmax in sampling for decode speedup | 2026-03-13 | kernel-fusion, pipeline-stages | reduction |
| [#20094](../sources/prs/sglang/PR-20094.md) | [diffusion] fix bug of copy_if | 2026-03-07 |  |  |
| [#20012](../sources/prs/sglang/PR-20012.md) | [JIT Kernel] Reland NVFP4 kernels to JIT | 2026-03-06 | swizzling | attention, fp4, gemm |
| [#19945](../sources/prs/sglang/PR-19945.md) | [AMD] Tilelang sparse fwd for dsv32 mi355/mi300 | 2026-03-05 | pipeline-stages | attention, gemm, sort |
| [#19725](../sources/prs/sglang/PR-19725.md) | [SGLang-Diffusion] Fix custom op fake impl missing eps default for torch.compile | 2026-03-03 | pipeline-stages | flash-attention |
| [#19794](../sources/prs/sglang/PR-19794.md) | Add compile-time 256-bit vector guard for pre-Blackwell | 2026-03-03 |  |  |
| [#19652](../sources/prs/sglang/PR-19652.md) | [Feature] NVFP4 Marlin fallback for non-Blackwell GPUs (SM75+) | 2026-03-02 |  | fp4, fp8, gemm |
| [#19549](../sources/prs/sglang/PR-19549.md) | [diffusion][llm] macOS support | 2026-02-28 | pipeline-stages | attention, quantization |
| [#19437](../sources/prs/sglang/PR-19437.md) | [Kernel Slimming] Migrate NVFP4 kernels to JIT | 2026-02-26 | swizzling | fp4, gemm, moe |
| [#19148](../sources/prs/sglang/PR-19148.md) | [DeepSeek-V3.2][JIT-kernel] Support nsa fuse store indexer k cache | 2026-02-22 |  | attention, fp8, quantization |
| [#19059](../sources/prs/sglang/PR-19059.md) | [jit_kernel] Add fused_qknorm_rope JIT kernel | 2026-02-20 |  | moe |
| [#18762](../sources/prs/sglang/PR-18762.md) | [diffusion] Diffusion norm fusion for z-image | 2026-02-13 |  | attention |
| [#18488](../sources/prs/sglang/PR-18488.md) | Tilelang sparse decode fwd for dsv32 mi355 | 2026-02-09 |  | attention, gemm, reduction |
| [#18496](../sources/prs/sglang/PR-18496.md) | [FIX] Correct JIT kernel compilation on newer GPUs with outdated driver metadata. | 2026-02-09 |  |  |
| [#18442](../sources/prs/sglang/PR-18442.md) | feat: add FA4 SM90 paged KV decode support & update attention docs | 2026-02-08 |  | attention, flash-attention, fp4 |
| [#18311](../sources/prs/sglang/PR-18311.md) | [Hicache & JIT_kernel] Support page first layout  & mla jit kernel | 2026-02-05 |  | mla |
| [#18073](../sources/prs/sglang/PR-18073.md) | [Diffsuion & JIT_kernel] QKNorm cross heads kernel | 2026-02-01 |  | reduction |
| [#17889](../sources/prs/sglang/PR-17889.md) | [Move sgl-kernel Kernel to JIT] Add JIT concat MLA kernels | 2026-01-28 |  | mla |
| [#17838](../sources/prs/sglang/PR-17838.md) | Feature/support longcat flash lite | 2026-01-27 |  |  |
| [#17554](../sources/prs/sglang/PR-17554.md) | Kernel: optimize decoding metadata in NSA multi-spec backend with fused kernels | 2026-01-22 | kernel-fusion | attention, fp8, sparse-attention |
| [#17449](../sources/prs/sglang/PR-17449.md) | Add mxfp8 support for online quantization, Triton dense linear, and CUTLASS MoE | 2026-01-21 |  | block-scale, fp4, fp8 |
| [#17353](../sources/prs/sglang/PR-17353.md) | Move fa4 from sgl-kernel to jit kernel | 2026-01-19 | double-buffering, pipeline-stages, swizzling | attention, flash-attention, fp8 |
| [#16723](../sources/prs/sglang/PR-16723.md) | [Rework] Add SwapAB Optimization for triton fused_moe_kernel on SM90. | 2026-01-08 |  | moe |
| [#16162](../sources/prs/sglang/PR-16162.md) | [Feature] add aligned_vector type for JIT kernel | 2025-12-30 |  |  |
| [#16043](../sources/prs/sglang/PR-16043.md) | optimize get_topk_ragged by fusing get k and k_scale triton kernel | 2025-12-29 |  | attention |
| [#15888](../sources/prs/sglang/PR-15888.md) | [diffusion] model: support TurboWan2.1-T2V-1.3B/14B SLA | 2025-12-26 |  | attention, sparse-attention, topk |
| [#15835](../sources/prs/sglang/PR-15835.md) | [Feature] JIT Fused QK norm + qk norm clean up | 2025-12-25 | persistent-kernel |  |
| [#15836](../sources/prs/sglang/PR-15836.md) | [JIT kernel] Apply jit per_tensor_quant_fp8 kernel | 2025-12-25 |  | gemm |
| [#15712](../sources/prs/sglang/PR-15712.md) | Add SwapAB Optimization for triton fused_moe_kernel on SM90. | 2025-12-24 |  | attention, fp8, moe |
| [#15631](../sources/prs/sglang/PR-15631.md) | [jit-kernel] Add CuTe DSL GDN Decode Kernel | 2025-12-22 |  | sort |
| [#15522](../sources/prs/sglang/PR-15522.md) | Optimize FP8 MLA KV cache writes with Triton kernel | 2025-12-20 |  | attention, fp8, mla |
| [#15539](../sources/prs/sglang/PR-15539.md) | MoE: Skip SiLU/GELU activation for masked experts | 2025-12-20 |  | moe |
| [#15471](../sources/prs/sglang/PR-15471.md) | [sgl-kernel][6/7]Support Expert Specialization Grouped GEMM | 2025-12-19 |  | fp8, gemm, grouped-gemm |
| [#15306](../sources/prs/sglang/PR-15306.md) | Fix warp illegal instruction in kimi k2 thinking PCG | 2025-12-17 |  | moe, topk |
| [#15141](../sources/prs/sglang/PR-15141.md) | [sgl-kernel][1/2] Fused qk_norm_rope for GLM4.6 | 2025-12-15 |  | attention, fp8, moe |
| [#15182](../sources/prs/sglang/PR-15182.md) | [NVIDIA] upstream FA4 | 2025-12-15 |  | attention, flash-attention, tma |
| [#14717](../sources/prs/sglang/PR-14717.md) | [diffusion] kernel fusion: gated residual layernorm scale shift and layernorm scale shift kernel fusion for Qwen-Image, WAN and HunyuanVideo | 2025-12-09 | kernel-fusion, pipeline-stages |  |
| [#14640](../sources/prs/sglang/PR-14640.md) | [sgl-kernel][Feat][B200][2/N] Support MXFP8 Grouped GEMM in Blackwell | 2025-12-08 |  | gemm, grouped-gemm |
| [#14311](../sources/prs/sglang/PR-14311.md) | [Fix] add block size logic for sm120 smem size | 2025-12-02 |  | attention |
| [#14224](../sources/prs/sglang/PR-14224.md) | [bug fix] fix ima with get_mla_kv_buffer_kernel overflow | 2025-12-01 |  | attention, sort |
| [#14122](../sources/prs/sglang/PR-14122.md) | Add new moe wna16 marlin gemm | 2025-11-29 |  | fp4, gemm, moe |
| [#14133](../sources/prs/sglang/PR-14133.md) | Opt moe align block size kernel | 2025-11-29 |  | moe, reduction, topk |
| [#14105](../sources/prs/sglang/PR-14105.md) | [LoRA][III] Add LoRA support for MoE layers and enable TP | 2025-11-28 |  | gemm, moe, quantization |
| [#13959](../sources/prs/sglang/PR-13959.md) | [DeepSeek v3.2] opt Context Parallelism: support fused moe, multi batch and fp8 kvcache | 2025-11-26 |  | attention, fp8, moe |
| [#13969](../sources/prs/sglang/PR-13969.md) | [kernel][moe] add moe topk fast | 2025-11-26 |  | moe, topk |
| [#13731](../sources/prs/sglang/PR-13731.md) | [sgl-kernel][Feat][B200][1/N]Support MXFP8 Grouped GEMM in Blackwell | 2025-11-21 |  | block-scale, gemm, grouped-gemm |
| [#13049](../sources/prs/sglang/PR-13049.md) | Support moe topk sigmoid kernel | 2025-11-11 |  | moe, reduction, topk |
| [#12581](../sources/prs/sglang/PR-12581.md) | [NVIDIA] Fix CUDA arch requirement in nvfp4 cast | 2025-11-04 |  | fp4, gemm, nvfp4 |
| [#12453](../sources/prs/sglang/PR-12453.md) | [Fix] `concat_mla_absorb_q_kernel` fails for long inputs | 2025-10-31 |  | attention |
| [#12080](../sources/prs/sglang/PR-12080.md) | [sgl-kernel][4/N]Support Expert Specialization Grouped GEMM | 2025-10-24 |  | gemm, grouped-gemm, moe |
| [#11737](../sources/prs/sglang/PR-11737.md) | support cutlass fp4 kernel in sm120 | 2025-10-17 | tile-scheduling | attention, fp4, gemm |
| [#11655](../sources/prs/sglang/PR-11655.md) | [DeepseekV32] Enable flashmla_prefill kernel with fp8 kvcache | 2025-10-15 | kernel-fusion | attention, fp4, fp8 |
| [#11664](../sources/prs/sglang/PR-11664.md) | Use trtllm_mla decode kernel for draft extend in speculative decoding | 2025-10-15 |  | attention, fp8 |
| [#11432](../sources/prs/sglang/PR-11432.md) | [sgl-kernel][1/N]Support Expert Specialization Grouped GEMM | 2025-10-10 |  | fp8, gemm, grouped-gemm |
| [#11287](../sources/prs/sglang/PR-11287.md) | [NVIDIA] Add new SMs support for Spark & Thor | 2025-10-07 |  | gemm |
| [#10714](../sources/prs/sglang/PR-10714.md) | Optimize cutlass int8 gemm kernel for large M on SM89 Ada GPU | 2025-09-21 |  | gemm |
| [#10543](../sources/prs/sglang/PR-10543.md) | [sgl-kernel] Optimize concat_mla_k kernel | 2025-09-17 |  | attention, fp8, quantization |
| [#10491](../sources/prs/sglang/PR-10491.md) | Update CUTLASS. Refine KernelSchedule for fp8 (grouped) gemm. | 2025-09-16 |  | fp8, gemm, moe |
| [#10426](../sources/prs/sglang/PR-10426.md) | Fix correction bias undefined behavior for nvfp4 models | 2025-09-14 |  | fp4, moe, nvfp4 |
| [#10101](../sources/prs/sglang/PR-10101.md) | Optimize nvfp4 block scaled gemm kernel when M is small. | 2025-09-06 |  | block-scale, fp4, gemm |
| [#10058](../sources/prs/sglang/PR-10058.md) | Disable kernel cutlass_mla_decode on SM103 | 2025-09-05 |  | attention |
| [#10078](../sources/prs/sglang/PR-10078.md) | feat: Add FP4 (E2M1) KV Cache Support with Quantization Utilities for MLA | 2025-09-05 |  | attention, fp4, fp8 |
| [#9969](../sources/prs/sglang/PR-9969.md) | CUTLASS fp8 blockwise gemm support of sm120 | 2025-09-03 |  | fp8, gemm |
| [#9807](../sources/prs/sglang/PR-9807.md) | Make fp4_quantize kernels work on sm103 | 2025-08-30 |  | fp4, gemm |
| [#9824](../sources/prs/sglang/PR-9824.md) | [Model] Support Meituan LongCat-Flash && LongCat-Flash-MTP | 2025-08-30 |  | moe |
| [#9789](../sources/prs/sglang/PR-9789.md) | Make sm100 fp8 kernels available on sm103 | 2025-08-29 |  | fp8, gemm, moe |
| [#9556](../sources/prs/sglang/PR-9556.md) | [NVIDIA] [2/N] Optimize `silu_and_mul_scaled_fp4_grouped_quant` perf | 2025-08-24 |  | fp4, gemm, nvfp4 |
| [#9559](../sources/prs/sglang/PR-9559.md) | Update CUTLASS 4.2 & Enable K-Major Scale Factor for SM90 FP8 Blockwise Group GEMM | 2025-08-24 |  | fp8, gemm, moe |
| [#9477](../sources/prs/sglang/PR-9477.md) | Optimize moe_sum_reduce_kernel | 2025-08-22 |  | moe |
| [#9403](../sources/prs/sglang/PR-9403.md) | [sgl-kernel] feat: Support sm120 cutlass fp8 gemm kernel | 2025-08-20 | tile-scheduling | fp8, gemm, quantization |
| [#9272](../sources/prs/sglang/PR-9272.md) | [fix]:  fix cutlass moe ut and and Opt H20 cutlass groupGemm performance | 2025-08-17 |  | gemm, moe |
| [#9200](../sources/prs/sglang/PR-9200.md) | [NVIDA] [1/N] Nvfp4 Masked Gemm: Add quant op for the flashinfer grouped gemm | 2025-08-14 | swizzling | fp4, fp8, gemm |
| [#8962](../sources/prs/sglang/PR-8962.md) | optimize: reduce shulffle and quantization overhead in cutlass_moe sm90 | 2025-08-08 |  | quantization, topk |
| [#8818](../sources/prs/sglang/PR-8818.md) | [Perf] Tunings for SM100 FP8 CUTLASS kernel | 2025-08-05 | tile-scheduling | fp8, gemm |
| [#8130](../sources/prs/sglang/PR-8130.md) | [sgl-kernel] Opt per_token_quant_fp8 with warp reduce | 2025-07-18 |  | fp8, gemm, quantization |
| [#8118](../sources/prs/sglang/PR-8118.md) | [feat] Support tp mode for DeepSeek-R1-W4AFP8 | 2025-07-17 |  | gemm, moe |
| [#8127](../sources/prs/sglang/PR-8127.md) | [Fix][Ready]Fix register spilling in cutlass nvfp4 gemm kernel on Blackwell | 2025-07-17 |  | fp4, gemm, nvfp4 |
| [#7884](../sources/prs/sglang/PR-7884.md) | [kernel] opt moe align block kernel by block/warp scan algorithm | 2025-07-09 |  | moe, scan, topk |
| [#7762](../sources/prs/sglang/PR-7762.md) | feat: support DeepSeek-R1-W4AFP8 model with ep-moe mode | 2025-07-04 |  | attention, fp8, gemm |
| [#7772](../sources/prs/sglang/PR-7772.md) | [1/n]: add cutlass W4A8 moe kernel for hopper architecture | 2025-07-04 |  | fp8, gemm, grouped-gemm |
| [#7627](../sources/prs/sglang/PR-7627.md) | Add dsv3 router gemm kernel | 2025-06-29 |  | fp4, gemm, reduction |
| [#7630](../sources/prs/sglang/PR-7630.md) | Add dsv3 fused a gemm to sgl-kernel | 2025-06-29 | swizzling | fp4, gemm, mbarrier |
| [#7437](../sources/prs/sglang/PR-7437.md) | Fuse sorted_token_ids padding to moe_align_block_size kernel | 2025-06-22 |  | moe, topk |
| [#7444](../sources/prs/sglang/PR-7444.md) | fix: fix apply_shuffle_mul_sum | 2025-06-22 |  | moe, topk |
| [#7278](../sources/prs/sglang/PR-7278.md) | Add CUTLASS FP8 Blockscale MoE kernel for Hopper architecture | 2025-06-17 |  | block-scale, fp8, gemm |
| [#7172](../sources/prs/sglang/PR-7172.md) | Support new DeepGEMM | 2025-06-14 |  | quantization |
| [#7057](../sources/prs/sglang/PR-7057.md) | Tiny fix cutlass_mla_get_workspace_size stub incorrect signature | 2025-06-10 |  | attention |
| [#6958](../sources/prs/sglang/PR-6958.md) | chore: upgrade flashinfer v0.2.6.post1 jit | 2025-06-08 |  |  |
| [#6970](../sources/prs/sglang/PR-6970.md) | Fuse routed scaling factor in deepseek | 2025-06-08 |  | moe |
| [#6916](../sources/prs/sglang/PR-6916.md) | Add a CUDA kernel for fusing mapping and weighted sum for MoE. | 2025-06-06 |  | gemm, moe, reduction |
| [#6919](../sources/prs/sglang/PR-6919.md) | [sgl-kernel] Add cuda kernel for moe_ep_silu_and_mul | 2025-06-06 |  | moe |
| [#6929](../sources/prs/sglang/PR-6929.md) | [perf][sgl-kernel] extend cutlass_mla_decode to support num_head < 128 | 2025-06-06 |  | attention, mla |
| [#6858](../sources/prs/sglang/PR-6858.md) | fix ep_moe_reorder kernel bugs | 2025-06-04 |  | moe, topk |
| [#6821](../sources/prs/sglang/PR-6821.md) | feat: integrate deepgemm into EPMoE | 2025-06-03 |  | attention, gemm, grouped-gemm |
| [#6837](../sources/prs/sglang/PR-6837.md) | [EP] Add cuda kernel for moe_ep_post_reorder | 2025-06-03 |  | moe, topk |
| [#6842](../sources/prs/sglang/PR-6842.md) | Fix AWQ Dequant and Weight Loading of deepseek v2 | 2025-06-03 |  | gemm |
| [#6782](../sources/prs/sglang/PR-6782.md) | Support token-level quantization for EP MoE | 2025-05-30 |  | moe, quantization |
| [#6736](../sources/prs/sglang/PR-6736.md) | Set `num_fused_shared_experts` as `num_shared_experts` when shared_experts fusion is not disabled | 2025-05-29 |  | moe, topk |
| [#6699](../sources/prs/sglang/PR-6699.md) | [EP] Add cuda kernel for moe_ep_pre_reorder | 2025-05-28 |  | moe, topk |
| [#6627](../sources/prs/sglang/PR-6627.md) | Refine pre_reorder_triton_kernel slightly to improve performance | 2025-05-26 |  | moe |
| [#6449](../sources/prs/sglang/PR-6449.md) | Fix bug of deepseek-v3 under DP+EP mode with large batchsize/seqlen | 2025-05-20 |  | attention, gemm, moe |
| [#6369](../sources/prs/sglang/PR-6369.md) | reduce torch.zeros overhead in moe align block size kernel | 2025-05-17 |  | moe |
| [#6336](../sources/prs/sglang/PR-6336.md) | Upgrade  CUTLASS 4.0 | 2025-05-15 |  | gemm |
| [#6093](../sources/prs/sglang/PR-6093.md) | [1/2] Add Kernel support for Cutlass based Fused FP4 MoE | 2025-05-07 | swizzling, tile-scheduling | block-scale, fp4, fp8 |
| [#6101](../sources/prs/sglang/PR-6101.md) | Cutlass MLA: Disable split kv due to https://github.com/NVIDIA/cutlass/issues/2274 | 2025-05-07 |  | attention, mla |
| [#6004](../sources/prs/sglang/PR-6004.md) | chore: upgrade cutlass 3.9.2 | 2025-05-04 |  | gemm |
| [#5820](../sources/prs/sglang/PR-5820.md) | cutlass 3.9 supported to improve fp8_blockwise_gemm | 2025-04-28 | persistent-kernel | fp8, gemm |
| [#5748](../sources/prs/sglang/PR-5748.md) | Fuse MLA set kv cache kernel | 2025-04-25 |  | mla |
| [#5694](../sources/prs/sglang/PR-5694.md) | [2/2] Add python wrapper for CUTLASS FP8 Blockscale MoE Kernel.  | 2025-04-24 |  | attention, block-scale, fp8 |
| [#5432](../sources/prs/sglang/PR-5432.md) | [perf] introduce deep gemm group_gemm_masked as bmm | 2025-04-15 |  | attention, fp8, gemm |
| [#5370](../sources/prs/sglang/PR-5370.md) | [perf] experimental enhance fp8 per-tensor quant | 2025-04-14 |  | fp8, mla, quantization |
| [#5331](../sources/prs/sglang/PR-5331.md) | fix: solve cu118 issue for cutlass mla | 2025-04-12 |  | attention, mla |
| [#5281](../sources/prs/sglang/PR-5281.md) | [1/2] Add FP8 Blockscale MoE CUTLASS kernel for Blackwell | 2025-04-11 | tile-scheduling | block-scale, fp8, gemm |
| [#5113](../sources/prs/sglang/PR-5113.md) | Support MHA with chunked prefix cache for DeepSeek chunked prefill | 2025-04-07 | pipeline-stages | attention, flash-attention, gemm |
| [#5142](../sources/prs/sglang/PR-5142.md) | Blackwell Cutlass MLA kernel | 2025-04-07 | tile-scheduling | attention, flash-attention, mla |
| [#5086](../sources/prs/sglang/PR-5086.md) | reduce moe_align_block_size_kernel small batch mode overhead | 2025-04-05 |  | moe, topk |
| [#4953](../sources/prs/sglang/PR-4953.md) | [Build] Fix cuda12.8 build error in nvfp4_scaled_mm_kernels.cu | 2025-03-31 |  | gemm |
| [#4887](../sources/prs/sglang/PR-4887.md) | Feat/support encoder model (like bert) | 2025-03-29 |  | attention |
| [#4706](../sources/prs/sglang/PR-4706.md) | support cmake for sgl-kernel | 2025-03-24 |  | attention, moe |
| [#4530](../sources/prs/sglang/PR-4530.md) | Add deepseek style fused moe group gate selection kernel | 2025-03-18 |  | moe, topk |
| [#4558](../sources/prs/sglang/PR-4558.md) | Support fp8 gemm for blackwell | 2025-03-18 | tile-scheduling | fp8, gemm |
| [#4278](../sources/prs/sglang/PR-4278.md) | Support Blackwell Block Scale FP8 Gemm | 2025-03-10 | persistent-kernel | block-scale, fp8, gemm |
| [#4231](../sources/prs/sglang/PR-4231.md) | fix per_token_group_quant_fp8 illegal memory when num_groups % 16 != 0 | 2025-03-09 |  | gemm |
| [#4199](../sources/prs/sglang/PR-4199.md) | linear support deepgemm | 2025-03-08 |  | gemm, quantization |
| [#4215](../sources/prs/sglang/PR-4215.md) | Accelerate FP8 CUDA Kernel by 20-28% | 2025-03-08 |  | fp8, gemm, moe |
| [#3888](../sources/prs/sglang/PR-3888.md) | [Feature] DeepSeek V3/R1 INT8 Quantization (channel-wise)  | 2025-02-26 |  | fp8, moe, quantization |
| [#3899](../sources/prs/sglang/PR-3899.md) | Support FP4 gemm (1/2) | 2025-02-26 | swizzling | fp4, fp8, gemm |
| [#3730](../sources/prs/sglang/PR-3730.md) | Feature DeepSeek V3/R1 INT8 Quantization (block-wise) | 2025-02-20 |  | fp8, moe, quantization |
| [#3529](../sources/prs/sglang/PR-3529.md) | integrate blockwise fp8 kernel | 2025-02-12 |  | fp8, quantization |
| [#3267](../sources/prs/sglang/PR-3267.md) | support blockwise fp8 matmul kernel | 2025-02-03 | persistent-kernel | fp8, gemm |
| [#3216](../sources/prs/sglang/PR-3216.md) | add tensorrt_llm common and cutlass_extensions as 3rdparty | 2025-01-30 | pipeline-stages, swizzling | fp8, gemm, quantization |
| [#3148](../sources/prs/sglang/PR-3148.md) | Apply sgl w8a8 fp8 kernel | 2025-01-26 |  | fp8, quantization, sort |
| [#3047](../sources/prs/sglang/PR-3047.md) | support w8a8 fp8 kernel with CUTLASS | 2025-01-22 | persistent-kernel, tile-scheduling | fp8, gemm, quantization |
| [#3035](../sources/prs/sglang/PR-3035.md) | Support sm90 Int8 gemm | 2025-01-21 | persistent-kernel, tile-scheduling | gemm, quantization |
| [#2752](../sources/prs/sglang/PR-2752.md) | Support cutlass Int8 gemm | 2025-01-06 |  | gemm, quantization |

<a id="tile-aitilelang"></a>
## tile-ai/tilelang
19 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#2216](../sources/prs/tilelang/PR-2216.md) | [TIR][IR] Update to use tirx | 2026-05-18 | pipeline-stages | attention |
| [#2218](../sources/prs/tilelang/PR-2218.md) | [Python] Drop Python 3.9 support | 2026-05-18 |  |  |
| [#2198](../sources/prs/tilelang/PR-2198.md) | [CUDA] Add native SM75 MMA GEMM support for FP16, INT8 and INT4 | 2026-05-13 |  | gemm |
| [#2153](../sources/prs/tilelang/PR-2153.md) | [codex] Split GEMM implementations by backend | 2026-05-06 |  | gemm, tcgen05, tmem |
| [#2112](../sources/prs/tilelang/PR-2112.md) | feat: auto-vectorize bf16/fp16 reduce with packed add2 intrinsics | 2026-04-28 |  | reduction |
| [#2073](../sources/prs/tilelang/PR-2073.md) | [CUDA] Improve int4 GEMM lowering and packed codegen support | 2026-04-21 |  | gemm |
| [#2063](../sources/prs/tilelang/PR-2063.md) | [CUDA] Support int4 `T.gemm` | 2026-04-19 | pipeline-stages | gemm |
| [#2048](../sources/prs/tilelang/PR-2048.md) | [Backend] Refactor gemm_sp | 2026-04-16 |  | gemm, wgmma |
| [#2003](../sources/prs/tilelang/PR-2003.md) | [Transform] Add InjectTcgen05Fence pass | 2026-03-31 |  | mbarrier, tcgen05, tmem |
| [#1981](../sources/prs/tilelang/PR-1981.md) | [Feature] Support TMA store in T.tma_copy() | 2026-03-27 | pipeline-stages | gemm, mbarrier, tma |
| [#1945](../sources/prs/tilelang/PR-1945.md) | [Feature] Block-scaled GEMM support for MXFP8 on Blackwell | 2026-03-18 | pipeline-stages, swizzling | block-scale, fp8, gemm |
| [#1909](../sources/prs/tilelang/PR-1909.md) | [Feature] Add Producer-Consumer Warp Specialization and T.tma_copy() API | 2026-03-07 | pipeline-stages, warp-specialization | gemm, mbarrier, tma |
| [#1882](../sources/prs/tilelang/PR-1882.md) | [Feature] 2-SM support for TMA, TMEM and TCGEN5MMA on Blackwell | 2026-02-27 | swizzling, warp-specialization | fp8, gemm, tcgen05 |
| [#1874](../sources/prs/tilelang/PR-1874.md) | [Feature] Support cluster launch, query, synchronization and barrier operations | 2026-02-24 |  |  |
| [#1866](../sources/prs/tilelang/PR-1866.md) | [CUDA] Support tcgen5mma gemm ts | 2026-02-22 | pipeline-stages | attention, flash-attention, gemm |
| [#1736](../sources/prs/tilelang/PR-1736.md) | Add swizzle layout detection and automatic merging for layout conflicts | 2026-01-27 | swizzling | attention, flash-attention, gemm |
| [#1667](../sources/prs/tilelang/PR-1667.md) | [Feature] Support `cp.reduce.async.bulk.tensor` | 2026-01-13 | swizzling | attention, flash-attention, tma |
| [#1327](../sources/prs/tilelang/PR-1327.md) | [Enhancement] add more dtype and fix mma.ws for fp16 for tcgen05 | 2025-11-24 | pipeline-stages | fp8, gemm, tcgen05 |
| [#1229](../sources/prs/tilelang/PR-1229.md) | [WIP] support more dtypes for tcgen05 | 2025-11-11 | pipeline-stages | fp8, gemm, tcgen05 |

<a id="vllm-projectvllm"></a>
## vllm-project/vllm
205 PRs

| PR | Title | Date | Techniques | Tags |
|-----|-------|------|------------|------|
| [#42899](../sources/prs/vllm/PR-42899.md) | add cutedsl dsv4 indexer fp8 kernel | 2026-05-17 |  | attention, fp4, fp8 |
| [#42849](../sources/prs/vllm/PR-42849.md) | [Perf] Add do_not_specialize in fused FP8 RoPE kernel | 2026-05-16 |  | attention, fp8 |
| [#42774](../sources/prs/vllm/PR-42774.md) | [Perf] Padded nvfp4 quant kernel to remove additional copy, 2.4%~5.7% e2e performance improvement | 2026-05-15 | swizzling | fp4, nvfp4, quantization |
| [#42663](../sources/prs/vllm/PR-42663.md) | [6/n] Migrate activation kernels, gptq, gguf, non cutlass w8a8 to libtorch stable ABI (continued) | 2026-05-14 |  | attention, fp8, moe |
| [#42527](../sources/prs/vllm/PR-42527.md) | [Kernel] Pack topk id/weights triton kernel | 2026-05-13 |  | topk |
| [#42236](../sources/prs/vllm/PR-42236.md) | [DSv4] Improved dequant gather K cache kernel | 2026-05-10 |  | attention, fp8, quantization |
| [#42153](../sources/prs/vllm/PR-42153.md) | [Perf] Use 2D-grid to eliminate divmod in W8W8 group quant | 2026-05-09 |  | fp8, quantization |
| [#42080](../sources/prs/vllm/PR-42080.md) | [feat] Add FP8 per-tensor Q scale support to Triton attention backend | 2026-05-08 |  | attention, fp8, quantization |
| [#41979](../sources/prs/vllm/PR-41979.md) | [MoE] Move various experts classes to fused_moe/experts/ | 2026-05-07 |  | moe, quantization |
| [#41986](../sources/prs/vllm/PR-41986.md) | [Bugfix] Add swiglu limits to deepgemm fp8 methods | 2026-05-07 |  | fp8, quantization |
| [#41326](../sources/prs/vllm/PR-41326.md) | Faster per-token fp8 group quant packed kernel for blackwell | 2026-04-30 | pipeline-stages | fp8, quantization, tma |
| [#41428](../sources/prs/vllm/PR-41428.md) | [DSv4] Improved fused Indexer Q quant kernel | 2026-04-30 |  | attention, block-scale, fp4 |
| [#41263](../sources/prs/vllm/PR-41263.md) | [DSV4]   Fuse norm and router for low latency scenario | 2026-04-29 |  | gemv, moe, reduction |
| [#40950](../sources/prs/vllm/PR-40950.md) | [DSV4] Add silu clamp limit to shared expert | 2026-04-27 |  |  |
| [#40392](../sources/prs/vllm/PR-40392.md) | [Performance][DSR1]: Fused RoPE+KVCache+q_concat for MLA | 2026-04-20 |  | attention, fp4, fp8 |
| [#40408](../sources/prs/vllm/PR-40408.md) | [Perf] Batch invariance with Cutlass fp8 support, 28.9% E2E latency improvement | 2026-04-20 |  | attention, fp8, quantization |
| [#40191](../sources/prs/vllm/PR-40191.md) | [Bugfix] Guard mxfp4_experts_quant bindings on ENABLE_NVFP4_SM100 | 2026-04-18 |  | fp4, nvfp4, quantization |
| [#40131](../sources/prs/vllm/PR-40131.md) | [Bugfix] moe lora align kernel grid | 2026-04-17 | pipeline-stages | moe, sort |
| [#39547](../sources/prs/vllm/PR-39547.md) | [Perf] Fuse Zero Initializer for FP8 DeepGemm Block Quant Kernel | 2026-04-10 |  | fp8, quantization, tma |
| [#39391](../sources/prs/vllm/PR-39391.md) | fix: clamp NaN/Inf in topk_softmax to prevent duplicate expert IDs | 2026-04-09 |  | fp8, moe, sort |
| [#39458](../sources/prs/vllm/PR-39458.md) | [MLA] Optimize mla indexer prepare uniform decode for MTP > 1 | 2026-04-09 |  | attention, fp4, mla |
| [#39306](../sources/prs/vllm/PR-39306.md) | Use CU_MEMCPY_SRC_ACCESS_ORDER_ANY for batch KV cache swaps | 2026-04-08 | pipeline-stages |  |
| [#38981](../sources/prs/vllm/PR-38981.md) | [Perf][GDN] Align TMA usage with upstream FLA | 2026-04-04 |  | attention, fp4, nvfp4 |
| [#38865](../sources/prs/vllm/PR-38865.md) | [Refactor] Improve indexer decode path metadata preparation | 2026-04-03 |  | attention, fp4, fp8 |
| [#38922](../sources/prs/vllm/PR-38922.md) | [Bugfix] Fix broken explicit unquantized kv cache dtype support | 2026-04-03 |  | attention, fp8, quantization |
| [#38460](../sources/prs/vllm/PR-38460.md) | [Perf] Batch KV cache swap copies via cuMemcpyBatchAsync | 2026-03-29 |  | reduction |
| [#38479](../sources/prs/vllm/PR-38479.md) | [Attention Backend] TurboQuant: 2-bit KV cache compression with 4x capacity | 2026-03-29 | kernel-fusion | attention, flash-attention, fp8 |
| [#38423](../sources/prs/vllm/PR-38423.md) | [NVIDIA] Bugfix NVFP4 DGX Spark and RTX50 | 2026-03-28 |  | fp4, gemm, grouped-gemm |
| [#38325](../sources/prs/vllm/PR-38325.md) | [Kernel] Add swapAB support for SM120 CUTLASS blockwise FP8 GEMM  | 2026-03-27 |  | fp8, gemm, quantization |
| [#38065](../sources/prs/vllm/PR-38065.md) | [Perf] FP8 FlashInfer Attn for ViT | 2026-03-25 |  | attention, fp4, fp8 |
| [#37948](../sources/prs/vllm/PR-37948.md) | [Perf] triton bilinear_pos_embed kernel for ViT | 2026-03-24 |  | fp4, nvfp4 |
| [#37970](../sources/prs/vllm/PR-37970.md) | [Kernel] Optimize SM120 CUTLASS blockwise FP8 GEMM | 2026-03-24 |  | fp8, gemm, quantization |
| [#37503](../sources/prs/vllm/PR-37503.md) | [4/n] Migrate FP4/W4A8 CUTLASS kernels to torch stable ABI | 2026-03-19 | swizzling | fp4, gemm, nvfp4 |
| [#37421](../sources/prs/vllm/PR-37421.md) | [Perf][Kernel] Persistent TopK scheduler: unified CUDAGraph-safe kernel with dynamic per-row dispatch - DeepSeek-V3.2 DSA decode | 2026-03-18 | double-buffering, persistent-kernel | fp4, fp8, nvfp4 |
| [#37463](../sources/prs/vllm/PR-37463.md) | [Kernel] Add MXFP4 W4A4 CUTLASS MoE kernel for SM100 | 2026-03-18 | swizzling, tile-scheduling | block-scale, fp4, gemm |
| [#37320](../sources/prs/vllm/PR-37320.md) | [Kernel] Add non-gated support for NVFP4 CUTLASS MoE | 2026-03-17 |  | block-scale, fp4, gemm |
| [#37332](../sources/prs/vllm/PR-37332.md) | Add nvfp4 support to reshape_and_cache_flash | 2026-03-17 | swizzling | block-scale, fp4, fp8 |
| [#37205](../sources/prs/vllm/PR-37205.md) | [Kernel] Add gpt-oss Router GEMM kernel | 2026-03-16 |  | gemm, mbarrier, moe |
| [#36982](../sources/prs/vllm/PR-36982.md) | [MTP][Sparse MLA] Take advantage of native MTP support in indexer when possible | 2026-03-13 |  | mla |
| [#36847](../sources/prs/vllm/PR-36847.md) | [Feat][Spec Decode] DFlash | 2026-03-12 | kernel-fusion | attention, flash-attention |
| [#36518](../sources/prs/vllm/PR-36518.md) | [Kernel] Fuse FP8 output quantization into merge_attn_states | 2026-03-09 |  | attention, fp8, quantization |
| [#36161](../sources/prs/vllm/PR-36161.md) | Add 320 dimension size support to MLA | 2026-03-05 |  | mla |
| [#35753](../sources/prs/vllm/PR-35753.md) | [Mamba] Add stochastic rounding support | 2026-03-02 |  |  |
| [#35777](../sources/prs/vllm/PR-35777.md) | [Kernel] Add fused_sigmoid_gating_delta_rule_update kernel for Qwen3 Next | 2026-03-02 | kernel-fusion | attention |
| [#35290](../sources/prs/vllm/PR-35290.md) | [Attention][Perf] Optimize cp_gather_and_upconvert_fp8_kv_cache - DeepSeek-v3.2 | 2026-02-25 |  | attention, fp4, fp8 |
| [#35161](../sources/prs/vllm/PR-35161.md) | [Bugfix] Fix expert_ids padding values in moe_align_block_size kernel | 2026-02-24 |  | moe |
| [#35210](../sources/prs/vllm/PR-35210.md) | [BugFix] Fix fp4 quant kernel on CUDA 12.8 | 2026-02-24 | swizzling | fp4, quantization |
| [#35219](../sources/prs/vllm/PR-35219.md) | [BUGFIX][Mamba][Qwen3.5] Zero freed SSM cache blocks on GPU | 2026-02-24 |  | attention, flash-attention, fp8 |
| [#35123](../sources/prs/vllm/PR-35123.md) | [Bugfix] Fix DSV3 kernels breaking _C and _moe_C on unsupported arches | 2026-02-23 |  | mla |
| [#34917](../sources/prs/vllm/PR-34917.md) | [Attention][Perf][Kernel] Replace torch.cat with vectorized CUDA kernel MLA query concat - DeepSeek-V3.2 | 2026-02-19 |  | attention, fp4, mla |
| [#34791](../sources/prs/vllm/PR-34791.md) | [Bugfix] Gate 256-bit instructions to CUDA 12.9+ | 2026-02-18 |  |  |
| [#34758](../sources/prs/vllm/PR-34758.md) | [Model Bash] DeepSeek R1 BF16 Min Latency QKV A GEMM (0.5% E2E Speedup) | 2026-02-17 | swizzling | fp4, gemm, mbarrier |
| [#34597](../sources/prs/vllm/PR-34597.md) | [Kernel] Add FP8 KV cache support to Triton MLA decode attention | 2026-02-16 |  | attention, fp8, mla |
| [#34556](../sources/prs/vllm/PR-34556.md) | [Quantization] add humming quantization kernel | 2026-02-14 | kernel-fusion | attention, block-scale, fp4 |
| [#34389](../sources/prs/vllm/PR-34389.md) | [Custom Ops] Add functional + out variant for scaled_fp4_quant | 2026-02-12 | swizzling | fp4, quantization |
| [#34448](../sources/prs/vllm/PR-34448.md) | [Kernel] Integrate SM100 MXFP8 blockscaled grouped MM and quant kernels | 2026-02-12 |  | block-scale, gemm, moe |
| [#34302](../sources/prs/vllm/PR-34302.md) | [ModelBash][DSV3] Add TRTLLM DSV3 Router GEMM kernel (6% B1 Speedup) | 2026-02-11 |  | fp4, gemm, moe |
| [#34206](../sources/prs/vllm/PR-34206.md) | [Kernel] Optimize grouped topk kernel | 2026-02-10 |  | moe, reduction, sort |
| [#33972](../sources/prs/vllm/PR-33972.md) | [Bugfix]fix output Nan/Inf in marlin if dtype=float16 | 2026-02-06 |  | fp4, gemm, moe |
| [#33529](../sources/prs/vllm/PR-33529.md) | Triton MLA perf fixes | 2026-02-02 |  | attention, mla |
| [#33517](../sources/prs/vllm/PR-33517.md) | [Kernel] Add enable_sm120_or_later for SM121 (DGX Spark) CUTLASS support | 2026-02-01 |  | fp4, fp8, gemm |
| [#33255](../sources/prs/vllm/PR-33255.md) | [Bugfix] Fix quant RMS norm fusion for quantization with TMA-aligned scales | 2026-01-28 |  | fp8, gemm, quantization |
| [#33291](../sources/prs/vllm/PR-33291.md) | [PERF] Change GDN Attention State Layout from [N, HV, K, V] to [N, HV, V, K] | 2026-01-28 |  | attention, fp4, fp8 |
| [#33022](../sources/prs/vllm/PR-33022.md) | [Kernel] Apply 256bit LDG/STG To Activation Kernels | 2026-01-25 |  |  |
| [#32873](../sources/prs/vllm/PR-32873.md) | [Performance] Tune Mamba selective scan kernel for B200 | 2026-01-22 |  | fp4, fp8, nvfp4 |
| [#32887](../sources/prs/vllm/PR-32887.md) | [Spec Decode] Unified Parallel Drafting | 2026-01-22 |  | attention, fp4, nvfp4 |
| [#32520](../sources/prs/vllm/PR-32520.md) | [Perf][Kernel] Optimize FP4 quantization kernels (SM100F) | 2026-01-17 | swizzling | fp4, gemm, moe |
| [#32195](../sources/prs/vllm/PR-32195.md) | Add TMA support to fused_moe_lora kernel | 2026-01-12 |  | moe, tma |
| [#31837](../sources/prs/vllm/PR-31837.md) | [Perf] Fuse stride preparation for NVFP4 cutlass_moe | 2026-01-06 |  | fp4, nvfp4, quantization |
| [#31380](../sources/prs/vllm/PR-31380.md) | [Bugfix][ROCm]Fix Qwen3-Next-80B-A3B-Thinking inference and optimize non-standard block size (544) support under rocm_atten | 2025-12-26 |  | attention, fp8 |
| [#31246](../sources/prs/vllm/PR-31246.md) | [Kernel] Add topk_sigmoid kernel | 2025-12-23 |  | moe, topk |
| [#30974](../sources/prs/vllm/PR-30974.md) | [Bugfix] Fix incorrect tiles creation for mm prefix triton attention | 2025-12-18 |  | attention |
| [#30887](../sources/prs/vllm/PR-30887.md) | [Bugfix] [Kernel] Triton attention kernels: mask out V blocks that fall outside sliding window | 2025-12-17 |  | attention |
| [#30897](../sources/prs/vllm/PR-30897.md) | [NVFP4][Perf] Tune NVFP4 input quant kernel for small batch size | 2025-12-17 | swizzling | fp4, moe, nvfp4 |
| [#30692](../sources/prs/vllm/PR-30692.md) | OffloadingConnector: Support kernel_block_size != block_size | 2025-12-15 |  | attention, mla |
| [#30286](../sources/prs/vllm/PR-30286.md) | [LoRA] Support Quantized Adapters | 2025-12-09 |  | fp8, gemm, grouped-gemm |
| [#30254](../sources/prs/vllm/PR-30254.md) | gptq marlin quantization support for fused moe with lora | 2025-12-08 |  | gemm, moe, quantization |
| [#30141](../sources/prs/vllm/PR-30141.md) | Add llmcompressor fp8 kv-cache quant (per-tensor and per-attn_head) | 2025-12-05 |  | attention, flash-attention, fp8 |
| [#29845](../sources/prs/vllm/PR-29845.md) | [SpecDecode] Simplified alternative padded-speculation acceptance rate fix | 2025-12-02 |  |  |
| [#29901](../sources/prs/vllm/PR-29901.md) | [Kernel][Quantization][MoE] add marlin kernel support for turing (sm75) | 2025-12-02 | pipeline-stages | fp4, fp8, moe |
| [#29642](../sources/prs/vllm/PR-29642.md) | [Kernel][MoE] optimize `moe_align_block_size` | 2025-11-28 |  | moe, topk |
| [#29691](../sources/prs/vllm/PR-29691.md) | [Kernel]Support W4A8 Grouped GEMM on Hopper | 2025-11-28 |  | block-scale, fp8, gemm |
| [#29354](../sources/prs/vllm/PR-29354.md) | Add unpermute-aware fused MoE path and small-batch fallback | 2025-11-24 |  | fp8, moe, quantization |
| [#29257](../sources/prs/vllm/PR-29257.md) | Lora MoE Align Improvements | 2025-11-23 |  | moe, scan, topk |
| [#29242](../sources/prs/vllm/PR-29242.md) | [Kernel] Add NVFP4 MoE CUTLASS support for SM120 | 2025-11-22 | tile-scheduling | fp4, gemm, moe |
| [#28840](../sources/prs/vllm/PR-28840.md) | bugfix: correct attn output with base 2 or e | 2025-11-17 |  | attention, mla |
| [#28775](../sources/prs/vllm/PR-28775.md) | [Model] Add support for openPangu moe model | 2025-11-15 |  | attention, fp8, moe |
| [#28358](../sources/prs/vllm/PR-28358.md) | [Performance][B200] silu_mul_quant: pack scales in int32 | 2025-11-09 |  | fp8, quantization |
| [#28124](../sources/prs/vllm/PR-28124.md) | [Perf][DeepSeek] Add sigmoid+bias fusion to fused_grouped_topk from TRTLLM | 2025-11-05 |  | fp4, moe, nvfp4 |
| [#27931](../sources/prs/vllm/PR-27931.md) | [Kernel] Optimize rms_norm kernel | 2025-11-01 |  |  |
| [#27883](../sources/prs/vllm/PR-27883.md) | [Performance] Fused blockwise quant RMS norm | 2025-10-31 | kernel-fusion | fp8, quantization |
| [#27532](../sources/prs/vllm/PR-27532.md) | [Attention] Use sparse prefill kernel for fp8 kv-cache in DeepSeek-v3.2 | 2025-10-26 |  | attention, fp8, mla |
| [#27284](../sources/prs/vllm/PR-27284.md) | [Perf] SM100 - add swap AB optimization to CUTLASS FP8 GEMM | 2025-10-21 |  | fp8, gemm, quantization |
| [#25843](../sources/prs/vllm/PR-25843.md) | Update launch_bounds_utils.h for correct compile on Multiple Cuda Arch - PTXAS out of range Warning | 2025-09-28 |  | fp4, moe, quantization |
| [#25774](../sources/prs/vllm/PR-25774.md) | Fuse RoPE and MLA KV-cache write | 2025-09-26 |  | fp8, mla, quantization |
| [#25193](../sources/prs/vllm/PR-25193.md) | [Compile] Fix Compile Warning for Ignoring `MIN_BLOCK_PER_SM` | 2025-09-18 |  | fp4, quantization |
| [#24833](../sources/prs/vllm/PR-24833.md) | [Bugfix] Fix accuracy issue for silu_mul + nvfp4 quant fusion kernel | 2025-09-14 |  | fp4, nvfp4, quantization |
| [#24722](../sources/prs/vllm/PR-24722.md) | [Kernel][Quantization] add w4a8 support for marlin kernel | 2025-09-12 | stream-k | fp4, fp8, moe |
| [#24673](../sources/prs/vllm/PR-24673.md) | [NVIDIA] Blackwell Family | 2025-09-11 |  | quantization |
| [#24440](../sources/prs/vllm/PR-24440.md) | [Transform] [Quantization] Add QuTLASS support to vLLM | 2025-09-08 | swizzling | block-scale, fp4, nvfp4 |
| [#24385](../sources/prs/vllm/PR-24385.md) | [Kernel] Support decode context parallelism on Blackwell with CUTLASS MLA | 2025-09-07 |  | attention, mla |
| [#23991](../sources/prs/vllm/PR-23991.md) | [Model] Add LongCat-Flash  | 2025-08-30 |  | moe |
| [#23972](../sources/prs/vllm/PR-23972.md) | [Kernel] Faster pre-processing time for W4A8 | 2025-08-29 |  | quantization |
| [#23791](../sources/prs/vllm/PR-23791.md) | [Kernel] cuda kernels for upcoming decode context parallel feature | 2025-08-28 |  | fp8 |
| [#23727](../sources/prs/vllm/PR-23727.md) | [Bugfix][Misc] Fix silu_and_mul_nvfp4_quant issue and extract common utils for nvfp4 kernel source files | 2025-08-27 |  | fp4, fp8, moe |
| [#23734](../sources/prs/vllm/PR-23734.md) | [Feature] Support Decode Context Parallel (DCP) for MLA | 2025-08-27 |  | mla |
| [#23660](../sources/prs/vllm/PR-23660.md) | [Compile] Fix Compile Warning for `w4a8_mm_entry.cu` | 2025-08-26 |  | quantization |
| [#23671](../sources/prs/vllm/PR-23671.md) | [NVIDIA] Support SiluMul + NVFP4 quant fusion | 2025-08-26 |  | fp4, fp8, gemm |
| [#23424](../sources/prs/vllm/PR-23424.md) | [Bugfix] Fixing division by zero in triton_attn if query_heads/kv_heads > 16  | 2025-08-22 |  | attention, flash-attention |
| [#23265](../sources/prs/vllm/PR-23265.md) | [Perf] Small optimizations for silu_mul_fp8_quant_deep_gemm | 2025-08-20 |  | fp8 |
| [#23274](../sources/prs/vllm/PR-23274.md) | [Kernel] Add fused grouped_topk kernel for MoE | 2025-08-20 | pipeline-stages | moe, reduction, sort |
| [#23280](../sources/prs/vllm/PR-23280.md) | [Perf] Use upstream CUTLASS for SM90 Block FP8 kernel | 2025-08-20 |  | block-scale, fp8, gemm |
| [#23287](../sources/prs/vllm/PR-23287.md) | [Compile] Fix Compile Warning SM100 Cutlass MLA | 2025-08-20 |  | attention, flash-attention, mla |
| [#23174](../sources/prs/vllm/PR-23174.md) | Optimize input preparation for FlashInfer [2/N] | 2025-08-19 |  | attention |
| [#23198](../sources/prs/vllm/PR-23198.md) | [kernel] Support W4A8 on Hopper | 2025-08-19 |  | fp8, gemm, quantization |
| [#23045](../sources/prs/vllm/PR-23045.md) | [Kernel] CUTLASS MoE FP8: Integrate cuda moe permute/unpermute | 2025-08-17 |  | block-scale, fp4, fp8 |
| [#22991](../sources/prs/vllm/PR-22991.md) | [Fix] enable swap_ab for pplx problem size computation | 2025-08-15 |  | gemm, moe, quantization |
| [#22368](../sources/prs/vllm/PR-22368.md) | [BugFix] Fix triton compile error in `kernel_unified_attention_2/3d` caused by attention sinks | 2025-08-06 |  | attention |
| [#22399](../sources/prs/vllm/PR-22399.md) | [Bug] Fix B200 DeepGEMM E8M0 Accuracy Issue | 2025-08-06 |  | fp8, quantization |
| [#22222](../sources/prs/vllm/PR-22222.md) | Fp8 paged attention update | 2025-08-05 |  | attention, fp4, fp8 |
| [#22131](../sources/prs/vllm/PR-22131.md) | [Kernel] Add support for block FP8 on SM120 (NVIDIA 5090 and RTX PRO 6000) | 2025-08-02 | tile-scheduling | fp8, gemm, quantization |
| [#21556](../sources/prs/vllm/PR-21556.md) | [Kernel] Improve machete memory bound perf | 2025-07-24 |  | quantization, tma |
| [#21465](../sources/prs/vllm/PR-21465.md) | [Bug] Fix Compressed Tensor NVFP4 `cutlass_fp4_group_mm` illegal memory access | 2025-07-23 |  | block-scale, fp4, moe |
| [#21309](../sources/prs/vllm/PR-21309.md) | Support CUTLASS NVFP4 (w4a4) for Blackwell Geforce GPUs (SM120) | 2025-07-21 | swizzling | fp4, gemm, nvfp4 |
| [#21249](../sources/prs/vllm/PR-21249.md) | [v1] - Mamba1 Attention Metadata | 2025-07-20 |  | attention |
| [#21229](../sources/prs/vllm/PR-21229.md) | [Feature][Kernel]FusedMoE LoRA | 2025-07-19 |  | moe, quantization |
| [#21197](../sources/prs/vllm/PR-21197.md) | [Kernel] Enable Hybrid Model Support in Triton Unified Attention Kernel | 2025-07-18 |  | attention, flash-attention, fp8 |
| [#21116](../sources/prs/vllm/PR-21116.md) | [perf] Add fused MLA QKV + strided layernorm | 2025-07-17 |  | fp8, mla, quantization |
| [#21083](../sources/prs/vllm/PR-21083.md) | [Perf] Cuda Kernel for Per Token Group Quant | 2025-07-16 |  | fp8, gemm, quantization |
| [#20903](../sources/prs/vllm/PR-20903.md) | [Kernel] DeepGemm MoE : Integrate triton permute / unpermute kernels  | 2025-07-14 |  | fp8, gemm, grouped-gemm |
| [#20911](../sources/prs/vllm/PR-20911.md) | [Perf] Add swap_ab to SM90 FP8 non-block CUTLASS moe grouped gemm | 2025-07-14 |  | block-scale, fp8, gemm |
| [#20833](../sources/prs/vllm/PR-20833.md) | [Bug] Fix DeepGemm for EP low latency case | 2025-07-11 |  | fp8 |
| [#20841](../sources/prs/vllm/PR-20841.md) | [Perf] Use Triton instead of Torch for DeepGEMM Per Token Group Quant | 2025-07-11 |  | fp8, quantization |
| [#20762](../sources/prs/vllm/PR-20762.md) | [Performance] Performance improvements in non-blockwise fp8 CUTLASS MoE | 2025-07-10 |  | fp8, moe, topk |
| [#20769](../sources/prs/vllm/PR-20769.md) | SM100 Cutlass MLA decode with unrestricted num_heads (< 128) for DeepSeek TP | 2025-07-10 | tile-scheduling | attention, flash-attention, mla |
| [#20781](../sources/prs/vllm/PR-20781.md) | [fix]: disable cutlass block scaled group gemm for EP | 2025-07-10 |  | block-scale, gemm, moe |
| [#20447](../sources/prs/vllm/PR-20447.md) | [feat]: add SM100 support for cutlass FP8 groupGEMM | 2025-07-03 |  | fp8, gemm, moe |
| [#20396](../sources/prs/vllm/PR-20396.md) | [Kernel] SM90 CUTLASS FP8 GEMM: add support for swap AB + kernel tuning | 2025-07-02 | persistent-kernel | fp8, gemm, quantization |
| [#20308](../sources/prs/vllm/PR-20308.md) | [Kernel] Optimize Prefill Attention in Unified Triton Attention Kernel | 2025-07-01 |  | attention, flash-attention |
| [#20324](../sources/prs/vllm/PR-20324.md) | [Kernel][Bugfix] Fixup some warnings in nvfp4_blockwise_moe when CUDA < 12.8 | 2025-07-01 |  | fp4, quantization |
| [#20166](../sources/prs/vllm/PR-20166.md) | [Bugfix] Fix topk_ids indices_type for CUTLASS w8a8 FP8 MoE | 2025-06-27 |  | fp8, moe, quantization |
| [#20141](../sources/prs/vllm/PR-20141.md) | [Bugfix] Fix some narrowing conversion warnings | 2025-06-26 |  | attention, fp4, mla |
| [#20087](../sources/prs/vllm/PR-20087.md) |  [Feature] Integrate SM100 DeepGEMM support | 2025-06-25 |  | block-scale, fp8, gemm |
| [#20016](../sources/prs/vllm/PR-20016.md) | Enable V1 for Hybrid SSM/Attention Models | 2025-06-24 |  | attention |
| [#19757](../sources/prs/vllm/PR-19757.md) | [feat]: CUTLASS block scaled group gemm for SM100 | 2025-06-17 |  | block-scale, gemm, moe |
| [#19566](../sources/prs/vllm/PR-19566.md) | [Perf] Further tunings for SM100 FP8 CUTLASS kernel | 2025-06-12 |  | fp8, gemm, quantization |
| [#19500](../sources/prs/vllm/PR-19500.md) | [Hardware][NVIDIA][kernel] Fp4 MOE quant kernel optimization | 2025-06-11 |  | fp4, gemm, moe |
| [#19110](../sources/prs/vllm/PR-19110.md) | [Hardware][NVIDIA] FP4 MoE kernel optimization | 2025-06-03 |  | block-scale, fp4, moe |
| [#18807](../sources/prs/vllm/PR-18807.md) | [BugFix] FA2 MLA Accuracy Issue | 2025-05-28 |  | attention, mla |
| [#18864](../sources/prs/vllm/PR-18864.md) | [Kernel] Enable fp8 support for pplx and BatchedTritonExperts. | 2025-05-28 |  | fp8, gemm, quantization |
| [#18762](../sources/prs/vllm/PR-18762.md) | [Kernel] Integrate CUTLASS MoE kernel with PPLX | 2025-05-27 |  | fp8, moe, quantization |
| [#18778](../sources/prs/vllm/PR-18778.md) | [Perf] Tunings for SM100 FP8 CUTLASS kernel | 2025-05-27 |  | fp8, gemm, quantization |
| [#18596](../sources/prs/vllm/PR-18596.md) | [Hardware][AMD] integrate aiter chunked prefill into vllm | 2025-05-23 |  | attention, fp8, quantization |
| [#18564](../sources/prs/vllm/PR-18564.md) | Sm100 blockwise fp8 swap ab | 2025-05-22 |  | fp8, gemm, quantization |
| [#18046](../sources/prs/vllm/PR-18046.md) | [Kernel] Have rotary embeddings support tensors | 2025-05-13 |  | mla |
| [#17687](../sources/prs/vllm/PR-17687.md) | [Kernel] fp4 marlin kernel | 2025-05-06 |  | fp4, fp8, moe |
| [#17280](../sources/prs/vllm/PR-17280.md) | [NVIDIA] Support Cutlass w8a8 FP8 for Blackwell Geforce GPUs (sm120) | 2025-04-28 |  | fp8, gemm, quantization |
| [#17139](../sources/prs/vllm/PR-17139.md) | [ROCm][FP8][Kernel] FP8 quantization fused into Custom Paged Attention | 2025-04-24 |  | attention, fp8, quantization |
| [#17082](../sources/prs/vllm/PR-17082.md) | Fix `numel()` downcast in vllm/csrc/moe/moe_align_sum_kernels.cu +2 | 2025-04-23 |  | moe |
| [#17004](../sources/prs/vllm/PR-17004.md) | [ROCm][Kernel][V1] Enable AMD Radeon GPU Custom Paged Attention on v1 | 2025-04-22 |  | attention, reduction |
| [#16828](../sources/prs/vllm/PR-16828.md) | [Kernel] Unified Triton kernel that doesn't distinguish between prefill + decode | 2025-04-18 |  | attention, flash-attention |
| [#16850](../sources/prs/vllm/PR-16850.md) | [Kernel] some optimizations for dense marlin and moe marlin | 2025-04-18 |  | fp8, moe, quantization |
| [#16861](../sources/prs/vllm/PR-16861.md) | [Kernel] Add expert_map support to Cutlass FP8 MOE | 2025-04-18 |  | fp8, moe, quantization |
| [#16780](../sources/prs/vllm/PR-16780.md) | [Kernel] GGUF MoeVec kernel | 2025-04-17 |  | moe, quantization, topk |
| [#16801](../sources/prs/vllm/PR-16801.md) | [BugFix] Accuracy fix for llama4 int4 - improperly casted scales | 2025-04-17 |  | moe |
| [#16605](../sources/prs/vllm/PR-16605.md) | Allocate kv_cache with stride order | 2025-04-14 |  | attention |
| [#16362](../sources/prs/vllm/PR-16362.md) | [Hardware/NVIDIA/Kernel] [Functional Enablement] [1/N] Enable nvidia/DeepSeek-R1-FP4 Model | 2025-04-09 | swizzling, tile-scheduling | block-scale, fp4, fp8 |
| [#16366](../sources/prs/vllm/PR-16366.md) | [Kernel] Support W8A8 channel-wise weights and per-token activations in triton fused_moe_kernel | 2025-04-09 |  | fp8, quantization |
| [#16173](../sources/prs/vllm/PR-16173.md) | [Kernel] support merge_attn_states CUDA kernel, 3x speedup | 2025-04-07 | pipeline-stages | attention, mla |
| [#16032](../sources/prs/vllm/PR-16032.md) | [NVIDIA] Support Cutlass MLA for Blackwell GPUs | 2025-04-03 | tile-scheduling | attention, flash-attention, fp4 |
| [#16034](../sources/prs/vllm/PR-16034.md) | [ROCM] Add gfx950 to the custom attention archs | 2025-04-03 |  | attention |
| [#15946](../sources/prs/vllm/PR-15946.md) | [Bugfix] fix use_atomic_add support of marlin kernel when using v1 engine | 2025-04-02 |  | quantization |
| [#15956](../sources/prs/vllm/PR-15956.md) | Modularize fused experts and integrate PPLX kernels | 2025-04-02 |  | moe, quantization, reduction |
| [#15720](../sources/prs/vllm/PR-15720.md) | [ROCM][KERNEL] Paged attention for V1 | 2025-03-28 |  | attention |
| [#15511](../sources/prs/vllm/PR-15511.md) | Use Cache Hinting for fused_moe kernel | 2025-03-26 |  |  |
| [#15456](../sources/prs/vllm/PR-15456.md) | [Kernel] Fix conflicting macro names for gguf kernels | 2025-03-25 |  | moe, quantization |
| [#14658](../sources/prs/vllm/PR-14658.md) | [Kernel] allow non-contiguous input for marlin kernel | 2025-03-12 |  | attention, mla, quantization |
| [#14613](../sources/prs/vllm/PR-14613.md) | [Kernel] GGUF MoE kernel | 2025-03-11 |  | moe, quantization |
| [#14568](../sources/prs/vllm/PR-14568.md) | permute/unpermute kernel for moe optimization | 2025-03-10 |  | gemm, grouped-gemm, moe |
| [#14447](../sources/prs/vllm/PR-14447.md) | [Kernel] moe wna16 marlin kernel | 2025-03-07 | pipeline-stages | mla, moe, quantization |
| [#14383](../sources/prs/vllm/PR-14383.md) | Add cutlass support for blackwell fp8 blockwise gemm | 2025-03-06 | tile-scheduling | fp8, gemm, quantization |
| [#14245](../sources/prs/vllm/PR-14245.md) | dynamic distpatch of fp8 kernels | 2025-03-05 |  | fp8, quantization |
| [#14138](../sources/prs/vllm/PR-14138.md) | [Kernel] optimize performance of gptq marlin kernel when n is small | 2025-03-03 |  | quantization |
| [#13972](../sources/prs/vllm/PR-13972.md) | [Kernel] CUTLASS grouped gemm fp8 MoE kernel | 2025-02-27 |  | fp8, gemm, grouped-gemm |
| [#13798](../sources/prs/vllm/PR-13798.md) | add cutlass support for blackwell fp8 gemm | 2025-02-25 |  | fp8, gemm, quantization |
| [#13718](../sources/prs/vllm/PR-13718.md) | [core] Perf improvement for DSv3 on AMD GPUs | 2025-02-23 |  | attention, mla, moe |
| [#13571](../sources/prs/vllm/PR-13571.md) | [NVIDIA] Support nvfp4 cutlass gemm | 2025-02-19 | swizzling | fp4, gemm, nvfp4 |
| [#13321](../sources/prs/vllm/PR-13321.md) | [Kernel] moe wna16 cuda kernel | 2025-02-15 |  | gemm, mla, moe |
| [#12978](../sources/prs/vllm/PR-12978.md) | [Kernel]Add streamK for block-quantized CUTLASS kernels | 2025-02-09 | persistent-kernel, stream-k, tile-scheduling | gemm, quantization |
| [#12931](../sources/prs/vllm/PR-12931.md) | [Misc][Kernel]: Add GPTQAllSpark Quantization | 2025-02-08 | double-buffering | gemm, quantization |
| [#12850](../sources/prs/vllm/PR-12850.md) | Optimize moe_align_block_size for deepseek_v3 | 2025-02-06 |  | moe |
| [#12777](../sources/prs/vllm/PR-12777.md) | [Kernel] Make rotary_embedding ops more flexible with input shape | 2025-02-05 |  | attention, mla |
| [#12784](../sources/prs/vllm/PR-12784.md) | [NVIDIA] Support nvfp4 quantization | 2025-02-05 |  | fp4, fp8, gemm |
| [#12676](../sources/prs/vllm/PR-12676.md) | [Perf] Mem align KV caches for CUDA devices (MLA perf improvement) | 2025-02-03 |  | attention, mla |
| [#12639](../sources/prs/vllm/PR-12639.md) | [Attention] MLA with chunked prefill | 2025-02-01 |  | attention, mla |
| [#12574](../sources/prs/vllm/PR-12574.md) | [Kernel] port sgl moe_align_block_size kernels | 2025-01-30 |  | moe |
| [#12583](../sources/prs/vllm/PR-12583.md) | Expert Parallelism (EP) Support for DeepSeek Models | 2025-01-30 |  | moe |
| [#12587](../sources/prs/vllm/PR-12587.md) | [Kernel][Quantization] Integrate block-quantized CUTLASS kernels for DeepSeekV3 | 2025-01-30 |  | quantization |
| [#12528](../sources/prs/vllm/PR-12528.md) | [Attention] MLA decode optimizations | 2025-01-28 |  | attention, fp8, mla |
| [#12348](../sources/prs/vllm/PR-12348.md) | [ROCm] Faster Custom Paged Attention kernels | 2025-01-23 |  | attention, fp8, gemm |
| [#12185](../sources/prs/vllm/PR-12185.md) | [Kernel] add triton fused moe kernel for gptq/awq | 2025-01-18 |  | moe, quantization, topk |
| [#11868](../sources/prs/vllm/PR-11868.md) | [Kernel] Update `cutlass_scaled_mm` to support 2d group (blockwise) scaling | 2025-01-08 | persistent-kernel, tile-scheduling | gemm, quantization |
| [#10995](../sources/prs/vllm/PR-10995.md) | [Kernel]: Cutlass 2:4 Sparsity + FP8/Int8 Quant Support | 2024-12-08 | persistent-kernel, tile-scheduling | fp8, gemm, quantization |
| [#7701](../sources/prs/vllm/PR-7701.md) | [Kernel] (2/N) Machete - Integrate into CompressedTensorsWNA16 and GPTQMarlin | 2024-08-20 |  | moe, quantization |
| [#7174](../sources/prs/vllm/PR-7174.md) | [Kernel] (1/N) Machete - Hopper Optimized Mixed Precision Linear Kernel  | 2024-08-05 | pipeline-stages, stream-k, tile-scheduling | fp8, gemm, quantization |

