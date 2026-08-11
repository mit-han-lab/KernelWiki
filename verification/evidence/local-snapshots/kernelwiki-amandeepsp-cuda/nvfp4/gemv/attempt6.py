from task import input_t, output_t
from typing import Type, Tuple
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.typing import Pointer
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr

from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm


@dsl_user_op
def atomic_add_fp32(a: float | Float32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> None:
    nvvm.atomicrmw(
        res=T.f32(), op=nvvm.AtomicOpKind.FADD, ptr=gmem_ptr.llvm_ptr, a=Float32(a).ir_value()
    )

@cute.jit
def scalar_to_ssa(a: cute.Numeric, dtype) -> cute.TensorSSA:
    """ Convert a scalar to a cute TensorSSA of shape (1,) and given dtype """
    vec = cute.make_rmem_tensor(1, dtype)
    vec[0] = a
    return vec.load()


def cdiv(a, b):
    return (a + b - 1) // b


class BlockScaledGemvHybrid:
    """
    Hybrid GEMV kernel with warp-cooperative processing and warp-shuffle reduction.
    
    Thread mapping (128 threads = 4 warps):
    - Each warp handles 1 M row, with 32 lanes splitting K tiles
    - Threads within a warp process K tiles in strided fashion
    - Warp shuffle reduces partial sums across lanes
    - No split-K (atomics are too expensive)
    """
    
    def __init__(
        self,
        b_m: int = 4,           # M elements per block (1 per warp)
        b_k: int = 64,          # K elements per tile
        sf_vec_size: int = 16,
        ab_dtype: Type[cute.Numeric] = cute.Float4E2M1FN,
        sf_dtype: Type[cute.Numeric] = cute.Float8E4M3FN,
        c_dtype: Type[cute.Numeric] = cute.Float16,
    ):
        self.sf_dtype = sf_dtype
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        
        self.b_m = b_m
        self.b_k = b_k
        self.sf_vec_size = sf_vec_size
        
        # Thread configuration
        self.threads_per_warp = 32
        self.warps_per_block = b_m  # 1 warp per M row
        self.num_threads_per_cta = self.threads_per_warp * self.warps_per_block
        
        self.n_padded = 128
        self.block_tiler_mnk = (b_m, 1, b_k)

    @cute.jit
    def __call__(
        self,
        a_ptr: Pointer,
        b_ptr: Pointer,
        sfa_ptr: Pointer,
        sfb_ptr: Pointer,
        c_ptr: Pointer,
        problem_size: Tuple[int, int, int, int],
    ):
        m, _, k, l = problem_size
        
        # A tensor with FP4 element type - CuTe handles sub-byte layout
        a_tensor = cute.make_tensor(
            a_ptr,
            cute.make_layout(
                (m, cute.assume(k, 32), l),
                stride=(cute.assume(k, 32), 1, cute.assume(m * k, 32)),
            ),
        )
        
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
        
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_tensor.shape, self.sf_vec_size
        )
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)

        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(
            b_tensor.shape, self.sf_vec_size
        )
        sfb_tensor = cute.make_tensor(sfb_ptr, sfb_layout)

        grid = (cdiv(m, self.b_m), 1, l)

        self._kernel(
            a_tensor,
            b_tensor,
            sfa_tensor,
            sfb_tensor,
            c_tensor,
        ).launch(grid=grid, block=(self.num_threads_per_cta, 1, 1))

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

        # Decompose thread ID into warp and lane
        warp_id = cute.arch.warp_idx()      # 0 to warps_per_block-1
        lane_id = cute.arch.lane_idx()      # 0-31
        
        # Each warp handles 1 M row
        m_idx = warp_id
        global_m = bidx * self.b_m + m_idx
        
        # Slice tensors to this batch
        mA = a[None, None, bidz]
        mB = b[None, None, bidz]
        mSFA = sfa[None, None, bidz]
        mSFB = sfb[None, None, bidz]
        mC = c[None, None, bidz]
        
        # Calculate number of K tiles
        k_tiles = cdiv(K, self.b_k)
        
        # Accumulator
        acc = cutlass.Float32(0.0)
        
        if global_m < M:
            # Each lane processes K tiles in strided fashion
            # Lane 0: tiles 0, 32, 64, ...
            # Lane 1: tiles 1, 33, 65, ...
            for k_tile in range(lane_id, k_tiles, self.threads_per_warp):
                k_start = k_tile * self.b_k
                
                if k_start < K:
                    # Tile coordinates
                    cta_coord = (bidx, 0, k_tile)
                    
                    # Get tiles for this M row and K tile
                    gA = cute.local_tile(mA, self.block_tiler_mnk, cta_coord, proj=(1, None, 1))
                    gSFA = cute.local_tile(mSFA, self.block_tiler_mnk, cta_coord, proj=(1, None, 1))
                    gB = cute.local_tile(mB, self.block_tiler_mnk, cta_coord, proj=(None, 1, 1))
                    gSFB = cute.local_tile(mSFB, self.block_tiler_mnk, cta_coord, proj=(None, 1, 1))
                    
                    # Select this M row's tile
                    tAgA = gA[m_idx, None]
                    tBgB = gB[0, None]
                    tAgSFA = gSFA[m_idx, None]
                    tBgSFB = gSFB[0, None]
                    
                    # Create register tensors
                    tArA = cute.make_rmem_tensor_like(tAgA, cutlass.Float16)
                    tBrB = cute.make_rmem_tensor_like(tBgB, cutlass.Float16)
                    tArSFA = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
                    tBrSFB = cute.make_rmem_tensor_like(tBgSFB, cutlass.Float32)
                    
                    # Load and convert FP4 to FP16 (CuTe handles sub-byte packing)
                    a_val_nvfp4 = tAgA.load()
                    b_val_nvfp4 = tBgB.load()
                    sfa_val_fp8 = tAgSFA.load()
                    sfb_val_fp8 = tBgSFB.load()
                    
                    a_val = a_val_nvfp4.to(cutlass.Float16)
                    b_val = b_val_nvfp4.to(cutlass.Float16)
                    sfa_val = sfa_val_fp8.to(cutlass.Float32)
                    sfb_val = sfb_val_fp8.to(cutlass.Float32)
                    
                    tArA.store(a_val)
                    tBrB.store(b_val)
                    tArSFA.store(sfa_val)
                    tBrSFB.store(sfb_val)
                    
                    # Compute element-wise products
                    tABrAB = cute.make_rmem_tensor_like(tAgA, cutlass.Float16)
                    tSFrSF = cute.make_rmem_tensor_like(tAgSFA, cutlass.Float32)
                    
                    tABrAB.store(tArA.load() * tBrB.load())
                    tSFrSF.store(tArSFA.load() * tBrSFB.load())
                    
                    # Accumulate within tile
                    for i in cutlass.range_constexpr(self.b_k):
                        global_k = k_start + i
                        if global_k < K:
                            acc = acc + cutlass.Float32(tABrAB[i]) * tSFrSF[i]
            
            # Warp-shuffle reduction across lanes
            acc = acc + cute.arch.shuffle_sync_down(acc, offset=16)
            acc = acc + cute.arch.shuffle_sync_down(acc, offset=8)
            acc = acc + cute.arch.shuffle_sync_down(acc, offset=4)
            acc = acc + cute.arch.shuffle_sync_down(acc, offset=2)
            acc = acc + cute.arch.shuffle_sync_down(acc, offset=1)
            
            # Only lane 0 writes the result (no atomics needed - each M row handled by one warp)
            if lane_id == 0:
                gC = cute.local_tile(mC, self.block_tiler_mnk, (bidx, 0, 0), proj=(1, 1, None))
                tCgC = gC[m_idx, None]
                tCgC = cute.make_tensor(tCgC.iterator, 1)
                out = scalar_to_ssa(acc, cute.Float32)
                tCgC.store(out.to(cutlass.Float16))


# Global cache for compiled kernel
_compiled_kernel_cache = None


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the hybrid block-scaled GEMV kernel with warp-shuffle reduction.
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    global _compiled_kernel_cache

    gemm = BlockScaledGemvHybrid(b_m=4, b_k=64)

    if _compiled_kernel_cache is None:
        a_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        c_ptr = make_ptr(gemm.c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        sfa_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
        sfb_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

        _compiled_kernel_cache = cute.compile(
            gemm, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0)
        )

        print(_compiled_kernel_cache.__ptx__)

    # Get dimensions - torch tensor has k/2 due to FP4 packing, multiply back
    m, k_packed, l = a.shape
    k = k_packed * 2  # Real K dimension
    n = 1

    a_ptr = make_ptr(
        gemm.ab_dtype, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    b_ptr = make_ptr(
        gemm.ab_dtype, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    c_ptr = make_ptr(
        gemm.c_dtype, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    sfa_ptr = make_ptr(
        gemm.sf_dtype, sfa_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )
    sfb_ptr = make_ptr(
        gemm.sf_dtype, sfb_permuted.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
    )

    _compiled_kernel_cache(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l))

    return c  # type: ignore
