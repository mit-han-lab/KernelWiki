# CUTLASS Kernels

Library of CUTLASS kernels targeting Large Language Models (LLM).

(07-11-24) The official version of FlashAttention-3 will be maintained at https://github.com/Dao-AILab/flash-attention.

We may upload some variants of the FA3 kernels to this repo from time to time for experimentation purposes, but we don't promise the same level of support here.

# Building 

1. Download CUTLASS following instructions from: https://github.com/NVIDIA/cutlass.
2. Modify the (hardcoded) path in the sample compile.sh to your CUTLASS directory.
3. Run the modified compile.sh as ./compile.sh.

# Running

1. While running the executable make sure to set NVIDIA_TF32_OVERRIDE=1 to enable TF32 mode for cuBLAS for SGEMM. Otherwise, cuBLAS uses float32.

# Notes

1. See README.md in sub-directories for more specific instructions.
