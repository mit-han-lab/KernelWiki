// provenance: derived from doc-ptx-isa-sm100 (PTX ISA block-scale mma.sync semantics, checked empirically); not upstream code
// Empirical probe of the per-lane operand layouts of the SM120 block-scaled warp MMAs:
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//   mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0
// Lanes that a hypothesis says are NOT read get poisoned values, so a PASS proves the selector semantics, not
// just the data path. Build: nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a (plain sm_120 rejects .kind::).
// Results (RTX PRO 6000 Blackwell, CUDA 13.0, 2026-10-01): see sm120-block-scaled-mma.md.
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>

__device__ __forceinline__ void mma_fp4_4x(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb, int tid_a, int tid_b) {
#define M4(TA, TB)                                                                                                       \
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {0," #TA "}, {%11}, {0," #TB "};"          \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                                        \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    switch (tid_a * 4 + tid_b) {
        case 0: M4(0, 0) break; case 1: M4(0, 1) break; case 2: M4(0, 2) break; case 3: M4(0, 3) break;
        case 4: M4(1, 0) break; case 5: M4(1, 1) break; case 6: M4(1, 2) break; default: M4(1, 3) break;
    }
#undef M4
}
__device__ __forceinline__ void mma_fp8_1x(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb, int bid_a, int tid_a, int bid_b, int tid_b) {
#define M8(BA, TA, BB, TB)                                                                                               \
    asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 " \
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {" #BA "," #TA "}, {%11}, {" #BB "," #TB "};" \
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])                                                        \
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
    switch (((bid_a * 2 + tid_a) * 4 + bid_b) * 2 + tid_b) {
        case 0: M8(0, 0, 0, 0) break; case 1: M8(0, 0, 0, 1) break; case 2: M8(0, 0, 1, 0) break; case 6: M8(0, 0, 3, 0) break;
        case 8: M8(0, 1, 0, 0) break; case 16: M8(1, 0, 0, 0) break; case 48: M8(3, 0, 0, 0) break; case 27: M8(1, 1, 1, 1) break;
        default: break;
    }
#undef M8
}

// fp4 case: A [16][64] e2m1 nibbles (one byte per element, value 0..15), B [8][64]; SA [16][4] e4m3 (k groups of 16), SB [8][4].
__global__ void k_fp4(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB, float* D, int tid_a, int tid_b) {
    int L = threadIdx.x, g = L / 4, q = L % 4;
    auto pack8 = [&](const uint8_t* row, int k0) { uint32_t v = 0; for (int i = 0; i < 8; i++) v |= (uint32_t)(row[k0 + i] & 15) << (4 * i); return v; };
    uint32_t a[4] = {pack8(A + g * 64, q * 8), pack8(A + (g + 8) * 64, q * 8), pack8(A + g * 64, 32 + q * 8), pack8(A + (g + 8) * 64, 32 + q * 8)};
    uint32_t b0 = pack8(B + g * 64, q * 8), b1 = pack8(B + g * 64, 32 + q * 8);
    // hypothesis: lanes with (q>>1)==tid_a supply the 4 k-group scales of row g + 8*(q&1); lanes with q==tid_b supply col g
    uint32_t sa = 0xEEEEEEEEu, sb = 0xEEEEEEEEu;  // poison (e4m3 0xEE = -1792) in unselected lanes
    if ((q >> 1) == tid_a) { sa = 0; for (int i = 0; i < 4; i++) sa |= (uint32_t)SA[(g + 8 * (q & 1)) * 4 + i] << (8 * i); }
    if (q == tid_b) { sb = 0; for (int i = 0; i < 4; i++) sb |= (uint32_t)SB[g * 4 + i] << (8 * i); }
    float d[4] = {0, 0, 0, 0};
    mma_fp4_4x(d, a, b0, b1, sa, sb, tid_a, tid_b);
    D[g * 8 + q * 2] = d[0]; D[g * 8 + q * 2 + 1] = d[1]; D[(g + 8) * 8 + q * 2] = d[2]; D[(g + 8) * 8 + q * 2 + 1] = d[3];
}
// fp8 case: A [16][32] e4m3 bytes, B [8][32]; SA [16][4] ue8m0 (4 consecutive k-steps' scales, byte-id selects), SB [8][4].
__global__ void k_fp8(const uint8_t* A, const uint8_t* B, const uint8_t* SA, const uint8_t* SB, float* D, int bid_a, int tid_a, int bid_b, int tid_b, int poison_b) {
    int L = threadIdx.x, g = L / 4, q = L % 4;
    uint32_t a[4] = {*(const uint32_t*)(A + g * 32 + q * 4), *(const uint32_t*)(A + (g + 8) * 32 + q * 4),
                     *(const uint32_t*)(A + g * 32 + 16 + q * 4), *(const uint32_t*)(A + (g + 8) * 32 + 16 + q * 4)};
    uint32_t b0 = *(const uint32_t*)(B + g * 32 + q * 4), b1 = *(const uint32_t*)(B + g * 32 + 16 + q * 4);
    uint32_t sa = 0xFEFEFEFEu, sb = 0xFEFEFEFEu;  // poison (ue8m0 0xFE = 2^127)
    if ((q >> 1) == tid_a) { sa = 0; for (int i = 0; i < 4; i++) sa |= (uint32_t)SA[(g + 8 * (q & 1)) * 4 + i] << (8 * i); }
    if (!poison_b || q == tid_b) { sb = 0; for (int i = 0; i < 4; i++) sb |= (uint32_t)SB[g * 4 + i] << (8 * i); }
    float d[4] = {0, 0, 0, 0};
    mma_fp8_1x(d, a, b0, b1, sa, sb, bid_a, tid_a, bid_b, tid_b);
    D[g * 8 + q * 2] = d[0]; D[g * 8 + q * 2 + 1] = d[1]; D[(g + 8) * 8 + q * 2] = d[2]; D[(g + 8) * 8 + q * 2 + 1] = d[3];
}

static float e2m1(uint8_t v) { static const float t[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6}; return (v & 8 ? -1.f : 1.f) * t[v & 7]; }
static float e4m3(uint8_t v) { __half_raw h = __nv_cvt_fp8_to_halfraw(v, __NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&h)); }
static float ue8m0(uint8_t e) { return ldexpf(1.f, (int)e - 127); }
template <typename T> static T* up(const std::vector<T>& h) { T* d; cudaMalloc(&d, h.size() * sizeof(T)); cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
static double maxrel(const float* D, const std::vector<float>& ref) { double m = 0; for (int i = 0; i < 128; i++) m = fmax(m, fabs(D[i] - ref[i]) / fmax(fabs(ref[i]), 1e-3)); return m; }

int main() {
    srand(1);
    float* dD; cudaMalloc(&dD, 512); float hD[128];
    // ---- fp4 / 4X
    {
        std::vector<uint8_t> A(16 * 64), B(8 * 64), SA(16 * 4), SB(8 * 4);
        for (auto& v : A) v = rand() & 15; for (auto& v : B) v = rand() & 15;
        for (int i = 0; i < 64; i++) SA[i] = 0x30 + (i % 13); for (int i = 0; i < 32; i++) SB[i] = 0x34 + (i % 7);  // e4m3 0x30=0.5 .. small positives
        std::vector<float> ref(128);
        for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) { double s = 0; for (int kg = 0; kg < 4; kg++) { double p = 0; for (int i = 0; i < 16; i++) p += e2m1(A[m * 64 + kg * 16 + i]) * e2m1(B[n * 64 + kg * 16 + i]); s += p * e4m3(SA[m * 4 + kg]) * e4m3(SB[n * 4 + kg]); } ref[m * 8 + n] = (float)s; }
        uint8_t *dA = up(A), *dB = up(B), *dSA = up(SA), *dSB = up(SB);
        for (int ta = 0; ta < 2; ta++) for (int tb = 0; tb < 4; tb++) {
            k_fp4<<<1, 32>>>(dA, dB, dSA, dSB, dD, ta, tb); cudaDeviceSynchronize(); cudaMemcpy(hD, dD, 512, cudaMemcpyDeviceToHost);
            printf("mxf4nvf4 4X  thread-id-a %d thread-id-b %d : maxrel %.2e %s\n", ta, tb, maxrel(hD, ref), maxrel(hD, ref) < 1e-5 ? "PASS" : "FAIL");
        }
    }
    // ---- fp8 / 1X
    {
        std::vector<uint8_t> A(16 * 32), B(8 * 32), SA(16 * 4), SB(8 * 4);
        for (auto& v : A) { v = rand() & 0x7f; if ((v & 0x78) == 0x78) v &= 0x77; } for (auto& v : B) { v = rand() & 0x7f; if ((v & 0x78) == 0x78) v &= 0x77; }
        for (int i = 0; i < 64; i++) SA[i] = 118 + (i % 11); for (int i = 0; i < 32; i++) SB[i] = 123 + (i % 7);
        uint8_t *dA = up(A), *dB = up(B), *dSA = up(SA), *dSB = up(SB);
        int sels[][4] = {{0,0,0,0},{0,0,0,1},{0,0,1,0},{0,0,3,0},{0,1,0,0},{1,0,0,0},{3,0,0,0},{1,1,1,1}};
        for (int poison_b = 0; poison_b < 2; poison_b++) for (auto& s : sels) {
            std::vector<float> ref(128);
            for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) { double p = 0; for (int i = 0; i < 32; i++) p += e4m3(A[m * 32 + i]) * e4m3(B[n * 32 + i]); ref[m * 8 + n] = (float)(p * ue8m0(SA[m * 4 + s[0]]) * ue8m0(SB[n * 4 + s[2]])); }
            k_fp8<<<1, 32>>>(dA, dB, dSA, dSB, dD, s[0], s[1], s[2], s[3], poison_b); cudaDeviceSynchronize(); cudaMemcpy(hD, dD, 512, cudaMemcpyDeviceToHost);
            printf("mxf8f6f4 1X  byte-id-a %d thread-id-a %d byte-id-b %d thread-id-b %d  B lanes q!=thread-id-b poisoned=%d : maxrel %.2e %s\n", s[0], s[1], s[2], s[3], poison_b, maxrel(hD, ref), maxrel(hD, ref) < 1e-5 ? "PASS" : "FAIL");
        }
    }
    return 0;
}
