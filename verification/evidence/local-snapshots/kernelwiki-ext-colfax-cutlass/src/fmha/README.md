# FMHA

NOTE: For the new FP8 and pipelined version, please go to /fmha-pipeline. This legacy version is preserved as companion code to our original paper.

An implementation of the FMHA algorithm based on FlashAttention-2, using CUTLASS/CuTe and built for Hopper (SM90) architecture. Explained in the companion paper "A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library" (https://research.colfax-intl.com/nvidia-hopper-flashattention-2/).

## Building and Running.

1. Download CUTLASS following instructions from: https://github.com/NVIDIA/cutlass.
2. Change the hardcoded paths in the compile.sh script to your CUTLASS directory.
3. To build the fmha kernel, run the compile.sh script. This produces the "fmha_forward" executable.
4. Run "fmha_forward --help" first to get list of command line arguments to be supplied for the executable.

## Building and Running for custom tile shapes.

1. Change the hardcoded paths in compile_run_all_config.sh to your CUTLASS directory.

2. Run as ./compile_run_all_config.sh [verify].  The script runs for pre-defined tile sizes for QBLK and KBLK, where QBLK is the QUERY_BLOCK_SIZE and KBLK is the KEY_BLOCK_SIZE. We report the best-performing versions in the "Results" section of the paper.

Note: you should see the warning "(C7512) Potential Performance Loss: wgmma.mma_async instructions are serialized due to insufficient register resources" appear in the case of 128 x 128. This is included for the sake of demonstrating a poor choice of tile sizes.

3. To run for different tiling configurations, you can also modify the compile.sh script by adding the compiler flags -DQBLKSIZE=$QBLK and -DKBLKSIZE=$KBLK with your desired values for QBLK and KBLK. Defaults are 64 x 64.

## NOTES:

1. Tested with CUDA 12.2, CUTLASS 3.3 and SM90A.
