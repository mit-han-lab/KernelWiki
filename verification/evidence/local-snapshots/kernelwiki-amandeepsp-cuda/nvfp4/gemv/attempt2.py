from task import input_t, output_t
from typing import Type, Tuple, Union

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.typing import Pointer
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.runtime import make_ptr


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

        grid = (cdiv(m, self.b_m), 1, l)

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

        cta_coord = (bidx, bidy, None)
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

            k_tiles = cute.size(gA, mode=[2])

            # --- Software Pipelining Setup ---
            
            # 1. Initialize Register Layouts for accumulation
            # We create these once to use repeatedly in the loop
            # Note: We just need the types/shapes, we can init from the first tile
            if k_tiles > 0:
                tAgA_first = gA[tidx, None, 0]
                tBgB_first = gB[0, None, 0]
                tAgSFA_first = gSFA[tidx, None, 0]
                tBgSFB_first = gSFB[0, None, 0]

                tArA = cute.make_rmem_tensor_like(tAgA_first, cutlass.Float16)
                tBrB = cute.make_rmem_tensor_like(tBgB_first, cutlass.Float16)
                tArSFA = cute.make_rmem_tensor_like(tAgSFA_first, cutlass.Float32)
                tBrSFB = cute.make_rmem_tensor_like(tBgSFB_first, cutlass.Float32)
                tABrAB = cute.make_rmem_tensor_like(tAgA_first, cutlass.Float16)
                tSFrSF = cute.make_rmem_tensor_like(tAgSFA_first, cutlass.Float32)

                # 2. Prologue: Load the 0-th tile immediately
                r_a_curr = tAgA_first.load()
                r_b_curr = tBgB_first.load()
                r_sfa_curr = tAgSFA_first.load()
                r_sfb_curr = tBgSFB_first.load()

                # Declare 'next' registers outside loop for scope visibility
                r_a_next = r_a_curr
                r_b_next = r_b_curr
                r_sfa_next = r_sfa_curr
                r_sfb_next = r_sfb_curr

                for k in cutlass.range(k_tiles, unroll_full=True):
                    # 3. Pipelining: Trigger Load for K+1 (prefetch)
                    # While math units are busy with K, memory units load K+1
                    next_k = k + 1
                    if next_k < k_tiles:
                        tAgA_next = gA[tidx, None, next_k]
                        tBgB_next = gB[0, None, next_k]
                        tAgSFA_next = gSFA[tidx, None, next_k]
                        tBgSFB_next = gSFB[0, None, next_k]

                        r_a_next = tAgA_next.load()
                        r_b_next = tBgB_next.load()
                        r_sfa_next = tAgSFA_next.load()
                        r_sfb_next = tBgSFB_next.load()

                    # 4. Compute on K (using `curr` registers)
                    # Convert loaded values to compute types
                    a_val = r_a_curr.to(cutlass.Float16)
                    b_val = r_b_curr.to(cutlass.Float16)
                    sfa_val = r_sfa_curr.to(cutlass.Float32)
                    sfb_val = r_sfb_curr.to(cutlass.Float32)

                    tArA.store(a_val)
                    tBrB.store(b_val)
                    tArSFA.store(sfa_val)
                    tBrSFB.store(sfb_val)

                    tABrAB.store(tArA.load() * tBrB.load())
                    tSFrSF.store(tArSFA.load() * tBrSFB.load())

                    for i in cutlass.range_constexpr(self.block_tiler_mnk[2]):
                        global_k = k * self.b_k + i
                        if global_k < K:
                            tCrC += tABrAB[i] * tSFrSF[i]

                    # 5. Shift Pipeline: Next becomes Current for the following iteration
                    if next_k < k_tiles:
                        r_a_curr = r_a_next
                        r_b_curr = r_b_next
                        r_sfa_curr = r_sfa_next
                        r_sfb_curr = r_sfb_next

            tCgC.store(tCrC.to(cute.Float16))


# Global cache for compiled kernel
_compiled_kernel_cache = None


def custom_kernel(data: input_t) -> output_t:
    """
    Execute the block-scaled GEMV kernel.
    """
    a, b, _, _, sfa_permuted, sfb_permuted, c = data

    global _compiled_kernel_cache

    stream = cutlass_torch.default_stream()

    gemm = BlockScaledGemv(block_tiler_mnk=(256, 1, 256), num_threads_per_cta=256)

    if _compiled_kernel_cache is None:
        a_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(gemm.ab_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        c_ptr = make_ptr(gemm.c_dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
        sfa_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)
        sfb_ptr = make_ptr(gemm.sf_dtype, 0, cute.AddressSpace.gmem, assumed_align=32)

        _compiled_kernel_cache = cute.compile(
            gemm, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (0, 0, 0, 0), stream
        )
        # print(_compiled_kernel_cache.__ptx__)

    m, k, l = a.shape
    k = k * 2
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

    _compiled_kernel_cache(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, (m, n, k, l), stream)

    return c  # type: ignore