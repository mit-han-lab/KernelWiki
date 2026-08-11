# FMHA Pipeline

Implementation of FMHA with software pipelining and optional warp specialization. Uses the CUTLASS Pipeline API. Both FP16 and FP8 precision formats for input tensors are supported.

## Building and Running with the scripts

1. Download CUTLASS 3.4 following instructions from: https://github.com/NVIDIA/cutlass.
2. Change the hardcoded paths in the 'compile_H64.sh' and 'compile_H128.sh' scripts to your CUTLASS directory, and run the scripts. We have preselected macros that we've found are optimal for FP8 precision with head dimension 64 and 128, respectively.
3. For more customizability, run the 'compile.sh' script. This accepts 7 arguments (as "-D$1 -D$2 -D$3 -D$4 -D$5 -D$6 -D$7").  If no flag is desired, simply input "NONE" as the argument.

## User-defined macros

You can tune a number of parameters through various macros. For integer flags:

1. EXECMODE={0,1,2}. The default pipelined version is 0. Setting to 1 enables warp-specialization, while setting to 2 enables our original non-pipelined version.
2. STAGECOUNT=#stages used during pipelining. Default is 3. For pipelined versions, this must be set to a value greater than 1. Since SMEM usage is determined dynamically, too large a value may lead to "Shared Memory Allocation Failed" error at runtime.
3. CLUSTERN=#threadblocks put in a cluster. Default is 1.
4. QBLKSIZE is the tile length bM along the first dimension for Q, and KBLKSIZE is the tile length bN along the first dimension for K and V. Default is 64 for both. We have tested for {64,128}x{64,128}. When CTA256 is enabled, QBLKSIZE should be 128.

For #ifdef flags:

1. GEMM2FP8 enables FP8 format for GEMM-II (P.V) in attention. Otherwise, the FP8 version is implemented as a 'hybrid' kernel where the Q and K tensors are FP8, but the V tensor remains FP16. Enabling GEMM2FP8 necessitates transposing V in memory as a pre-processing step prior to launching the kernel. Furthermore, the reported FLOPS with GEMM2FP8 enabled does not include the cost of this transpose as it is done offline.
2. CTA256 enables 256 threads (=2 warpgroups) per WGMMA.
3. QINRMEM enables the Q operand in RMEM for GEMM-I. This is always disabled for the non-pipelined version.
4. GEMM1FP16ACC enables FP16 accumulator for GEMM-I. Not recommended due to severely degraded accuracy of computation (we want float accumulator for softmax).
5. GEMM2FP16ACC enables FP16 accumulator for GEMM-II.

We also have the debugging flags COPYOUTMM0 and COPYOUTMI for validating intermediate steps of the computation. 

## Using the compile and run all script

The 'compile_run_all_config.sh' script accepts 7 arguments as follows:

1. $1 is meant to enable validation via "verify" or "verify-all".
3. $2 should be CTA256 or NONE.
4. $3 through $6 are additional optional flags.

The script then runs over predetermined possibilities for QBLKSIZE, KBLKSIZE, head sizes (64, 128, or 256), and precision type (FP16 or FP8).

## NOTES:

1. Tested with CUDA 12.2/12.3, CUTLASS 3.3/3.4 and SM90A.
2. A small change in the CUTLASS Pipeline API from 3.3 to 3.4 necessitates changes in the code for successful compilation. This version compiles with 3.4 and we have indicated the small changes required for use with 3.3 in the comments of the driver header files.
3. A choice of large tile size may lead to nvcc warnings related to register spilling. However, since we support head dimensions {64,128,256} and {FP16,FP8} within the same program, one should be observant as to which choice(s) of head dimension and precision type is triggering the warning. This is easily seen from the verbose ptxas output about register usage.
