from task import input_t, output_t
from typing import Type, Tuple, Union
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.typing import Pointer
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm, llvm

@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )


def cdiv(a, b):
    return (a + b - 1) // b


class BlockScaledGemv:
    def __init__(
        self,
        block_tiler_mnk: Tuple[int, int, int],
        sf_vec_size: int = 16,
        num_threads_per_cta=128,
        ab_dtype: Type[cute.Numeric] = cute.Float4E2M1FN,
        sf_dtype: Type[cute.Numeric] = cute.Float8E4M3FN,
        c_dtype: Type[cute.Numeric] = cute.Float16,
    ):
        self.sf_dtype = sf_dtype
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.b_m, self.b_n, self.b_k = block_tiler_mnk
        self.block_tiler_mnk = block_tiler_mnk
        self.sf_vec_size = sf_vec_size
        self.num_threads_per_cta = num_threads_per_cta

        self.n_padded = 128

    @cute.jit
    def __call__(
        self,
        a_ptr: Pointer,
        b_ptr: Pointer,
        sfa_ptr: Pointer,
        sfb_ptr: Pointer,
        c_ptr: Pointer,
        problem_size: Tuple[int, int, int, int],
        stream,
    ):
        m, _, k, l = problem_size
        # Create CuTe Tensor via pointer and problem size.
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        # We use n=128 to create the torch tensor to do fp4 computation via torch._scaled_mm
        # then copy torch tensor to cute tensor for cute customize kernel computation
        # therefore we need to ensure b_tensor has the right stride with this 128 padded size on n.
        n_padded_128 = 128
        b_tensor = cute.make_tensor(
            b_ptr,
            cute.make_layout(
                (n_padded_128, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(n_padded_128 * k, 32)),
            ),
        )
        c_tensor = cute.make_tensor(
            c_ptr, cute.make_layout((cute.assume(m, 32), 1, l), stride=(1, 1, m))
        )
        # Convert scale factor tensors to MMA layout
        # The layout matches Tensor Core requirements: (((32, 4), REST_M), ((SF_K, 4), REST_K), (1, REST_L))
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        grid = (cdiv(m, self.b_m), cdiv(k, self.b_k), l)

        self._kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
        ).launch(grid=grid, block=(self.num_threads_per_cta, 1, 1), stream=stream)

    @cute.kernel
    def _kernel(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        c: cute.Tensor,
    ):
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        M = cute.size(a, mode=[0])
        K = cute.size(a, mode=[1])

        mA = a[None, None, bidz]
        mB = b[None, None, bidz]
        mSFA = sfa[None, None, bidz]
        mSFB = sfb[None, None, bidz]
        mC = c[None, None, bidz]

        cta_coord = (bidx, 0, bidy)
        gA = cute.local_tile(mA, self.block_tiler_mnk, cta_coord, proj=(1, None, 1))
        gSFA = cute.local_tile(mSFA, self.block_tiler_mnk, cta_coord, proj=(1, None, 1))
        gB = cute.local_tile(mB, self.block_tiler_mnk, cta_coord, proj=(None, 1, 1))
        gSFB = cute.local_tile(mSFB, self.block_tiler_mnk, cta_coord, proj=(None, 1, 1))
        gC = cute.local_tile(mC, self.block_tiler_mnk, cta_coord, proj=(1, 1, None))

        tidx = cute.assume(tidx, 8)

        global_m = bidx * self.b_m + tidx

        if global_m < M:
            # Select output element corresponding to this thread and block indices
            tCgC = gC[tidx, None]
            tCgC = cute.make_tensor(tCgC.iterator, 1)
            tCrC = cute.zeros_like(tCgC, cutlass.Float32)

            tAgA = gA[tidx, None]
            tBgB = gB[0, None]
            tAgSFA = gSFA[tidx, None]
            tBgSFB = gSFB[0, None]

            tArA = cute.make_rmem_tensor_like(tAgA, cutlass.Float16)
            tBrB = cute.make_rmem_tensor_like(tBgB, cutlass.Float16)
            tArSFA = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
            tBrSFB = cute.make_rmem_tensor_like(tBgSFB, cutlass.Float32)

            tABrAB = cute.make_rmem_tensor_like(tAgA, cutlass.Float16)
            tSFrSF = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)

            # Load NVFP4 or FP8 values from global memory
            a_val_nvfp4 = tAgA.load()
            b_val_nvfp4 = tBgB.load()
            sfa_val_fp8 = tAgSFA.load()
            sfb_val_fp8 = tBgSFB.load()

            # Convert loaded values to float32 for computation (FFMA)
            a_val = a_val_nvfp4.to(cutlass.Float16)
            b_val = b_val_nvfp4.to(cutlass.Float16)
            sfa_val = sfa_val_fp8.to(cutlass.Float32)
            sfb_val = sfb_val_fp8.to(cutlass.Float32)

            # Store the converted values to RMEM CuTe tensors
            tArA.store(a_val)
            tBrB.store(b_val)
            tArSFA.store(sfa_val)
            tBrSFB.store(sfb_val)

            tABrAB.store(tArA.load() * tBrB.load())
            tSFrSF.store(tArSFA.load() * tBrSFB.load())

            for i in cutlass.range_constexpr(self.block_tiler_mnk[2]):
                global_k = bidy * self.b_k + i
                if global_k < K:
                    tCrC += tABrAB[i] * tSFrSF[i]

            atomic_add_fp32(tCrC[0], tCgC.iterator)


# Global cache for compiled kernel
_compiled_kernel_cache = None


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.

    This is the main entry point called by the evaluation framework.
    It converts PyTorch tensors to CuTe tensors, launches the kernel,
    and returns the result.

    Args:
        data: Tuple of (a, b, sfa_cpu, sfb_cpu, c) PyTorch tensors
            a: [m, k, l] - Input matrix in float4e2m1fn
            b: [1, k, l] - Input vector in float4e2m1fn
            sfa_cpu: [m, k, l] - Scale factors in float8_e4m3fn
            sfb_cpu: [1, k, l] - Scale factors in float8_e4m3fn
            sfa_permuted: [32, 4, rest_m, 4, rest_k, l] - Scale factors in float8_e4m3fn
            sfb_permuted: [32, 4, rest_n, 4, rest_k, l] - Scale factors in float8_e4m3fn
            c: [m, 1, l] - Output vector in float16

    Returns:
        Output tensor c with computed GEMV results
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    # Ensure kernel is compiled (will use cached version if available)
    # To avoid the compilation overhead, we compile the kernel once and cache it.
    global _compiled_kernel_cache

    stream = cutlass_torch.default_stream()

    gemm = BlockScaledGemv(block_tiler_mnk=(128, 128, 512), num_threads_per_cta=128)

    if _compiled_kernel_cache is None:
        # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
        a_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        # Use Float32 for output since atomic_add_fp32 writes float32
        c_ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
        sfa_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
        sfb_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

        # Compile the kernel
        _compiled_kernel_cache = cute.compile(
            gemm, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0), stream
        )

        print(_compiled_kernel_cache.__ptx__)

    # Get dimensions from MxKxL layout
    m, k, l = a.shape
    # Torch use e2m1_x2 data type, thus k is halved
    k = k * 2
    # GEMV N dimension is always 1
    n = 1

    # Create CuTe pointers for A/B/C/SFA/SFB via torch tensor data pointer
    a_ptr = make_ptr(
        gemm.ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        gemm.ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )

    c = c.to(torch.float32)
    c.zero_()  # Zero out before atomic adds

    # Use Float32 for output since atomic_add_fp32 writes float32
    c_ptr = make_ptr(
        cutlass.Float32, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfa_ptr = make_ptr(
        gemm.sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        gemm.sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    # Execute the compiled kernel
    _compiled_kernel_cache(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), stream)

    return c.to(torch.float16) # type: ignore