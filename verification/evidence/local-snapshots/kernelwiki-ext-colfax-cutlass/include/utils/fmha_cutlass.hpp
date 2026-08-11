#pragma once

#include <vector>

#include "cutlass/cutlass.h"

#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/host/gemm_complex.h"
#include "cutlass/util/tensor_view_io.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element, typename AccumType> class TestAttention {
public:
  //
  // Type definitions
  //
  static constexpr float kLog2e = float(1.4426950408889634074); // log_2(e) = M_LOG2E
  using ElementQ = Element;
  using ElementK = Element;
  using ElementS = Element;
  using ElementP = AccumType;
  using ElementAccumulator = AccumType;
  using ElementV = Element;
  using ElementO = Element;

  using ElementCompute = AccumType;

  using ElementNorm = AccumType;
  using ElementSum = AccumType;
  using ElementSoftmaxCompute = AccumType;

  using LayoutQ = cutlass::layout::RowMajor;
  using LayoutK = cutlass::layout::ColumnMajor;
  using LayoutP = cutlass::layout::RowMajor;
  using LayoutV = cutlass::layout::RowMajor;
  using LayoutO = cutlass::layout::RowMajor;
  using LayoutNorm = cutlass::layout::RowMajor;

  using MatrixCoord = typename LayoutP::TensorCoord;

private:
  bool help;
  bool error;
  bool reference_check;
  bool use_mask;
  bool causal;

  int alignment;
  int head_number;
  int batch_size;
  int head_size;
  int head_size_v;
  int seq_length;
  int seq_length_kv;
  int iterations;

  // alpha0, alpha1 and beta are fixed
  // in this multi-head attention example
  float alpha0;
  float alpha1;
  float beta;

  cutlass::gemm::GemmCoord problem0;
  cutlass::gemm::GemmCoord problem1;
  //

  int64_t ldq;
  int64_t ldk;
  int64_t ldp;
  int64_t ldv;
  int64_t ldo;

  int64_t elements_Q;
  int64_t elements_K;
  int64_t elements_V;
  int64_t elements_P;
  int64_t elements_O;
  int64_t elements_norm;

public:
  //
  // Methods
  //

  TestAttention(int _head_number, int _batch_size, int _head_size,
                int _seq_length, int _alignment = 1, bool use_mask = false,
                bool causal = false) {
    head_number = _head_number;
    batch_size = _batch_size;
    head_size = _head_size;
    seq_length = _seq_length;
    alignment = _alignment;

    head_size_v = head_size;
    seq_length_kv = seq_length;

    // problems belonging to the same batch share the same seq len
    int m_real = seq_length;
    int mkv_real = seq_length_kv;
    int m = (m_real + alignment - 1) / alignment * alignment;
    int mkv = (mkv_real + alignment - 1) / alignment * alignment;
    int k0 = head_size;
    int k1 = head_size_v;

    cutlass::gemm::GemmCoord _problem0(m, mkv, k0);
    cutlass::gemm::GemmCoord _problem1(m, k1, mkv);
    problem0 = _problem0;
    problem1 = _problem1;
  }

  int problem_count() const { return (head_number * batch_size); }

public:
  /// Initializes data structures
  void initialize() {

    //
    // Set scaling factor(s).
    //

    alpha0 = 1.0f / sqrt(float(head_size));
    alpha1 = 1.0f;
    beta = 0;

    //
    // Choose random problem sizes
    //

    // Create tensors in BMHK format, where
    // B = batch_size
    // M = sequence length
    // H = num_heads
    // K = embedding size per head

    elements_Q = problem0.m() * problem0.k() * head_number;
    elements_K = problem0.k() * problem0.n() * head_number;
    elements_P = problem0.m() * problem0.n();
    elements_V = problem1.k() * problem1.n() * head_number;
    elements_O = problem1.m() * problem1.n() * head_number;
    elements_norm = problem0.m();

    ldq = LayoutQ::packed({problem0.m(), head_number * problem0.k()}).stride(0);
    ldk = LayoutK::packed({head_number * problem0.k(), problem0.n()}).stride(0);
    ldp = LayoutP::packed({problem0.m(), problem0.n()}).stride(0);
    ldv = LayoutV::packed({problem1.k(), head_number * problem1.n()}).stride(0);
    ldo = LayoutO::packed({problem1.m(), head_number * problem1.n()}).stride(0);
  }

  /// Verifies the result using plain GEMM + Softmax.
  void compute(const ElementQ *Q, const ElementK *K, const ElementV *V,
               ElementS *S, ElementO *O, ElementNorm *norm, ElementSum *sum,
               bool usePow2 = false, bool usePreScaling = true) {

    LayoutQ layout_Q(ldq);
    LayoutK layout_K(ldk);
    LayoutP layout_P(ldp);
    LayoutV layout_V(ldv);
    LayoutV layout_O(ldo);

    LayoutNorm layout_norm(1);

    MatrixCoord extent_Q{problem0.m(), problem0.k()};
    MatrixCoord extent_K{problem0.k(), problem0.n()};
    MatrixCoord extent_P{problem0.m(), problem0.n()};
    MatrixCoord extent_V{problem1.k(), problem1.n()};
    MatrixCoord extent_O{problem1.m(), problem1.n()};
    MatrixCoord extent_norm{problem1.m(), 1};
    cutlass::DeviceAllocation<ElementP> softmaxP(layout_P.capacity(extent_P));
    cutlass::TensorView<ElementP, LayoutP> softmaxViewP(softmaxP.get(),
                                                        layout_P, extent_P);
    for (int64_t b = 0; b < batch_size; ++b) {

      for (int64_t h = 0; h < head_number; ++h) {

        auto offsetQ = Q + elements_Q * b + h * problem0.k();
        auto offsetK = K + elements_K * b + h * problem0.k();
        auto offsetV = V + elements_V * b + h * problem0.k();
        auto offsetO = O + elements_O * b + h * problem1.n();
        auto offsetS = S + elements_P * head_number * b + elements_P * h;
        auto offsetNorm =
            norm + elements_norm * head_number * b + elements_norm * h;
        auto offsetSum =
            sum + elements_norm * head_number * b + elements_norm * h;

        cutlass::TensorView<ElementQ, LayoutQ> view_Q(
            const_cast<ElementQ *>(offsetQ), layout_Q, extent_Q);
        cutlass::TensorView<ElementK, LayoutK> view_K(
            const_cast<ElementK *>(offsetK), layout_K, extent_K);
        cutlass::TensorView<ElementV, LayoutV> view_V(
            const_cast<ElementV *>(offsetV), layout_V, extent_V);
        cutlass::TensorView<ElementS, LayoutP> view_S(offsetS, layout_P,
                                                      extent_P);
        cutlass::TensorView<ElementO, LayoutO> view_O(offsetO, layout_O,
                                                      extent_O);
        cutlass::TensorView<ElementNorm, LayoutNorm> view_Norm_Ref(
            offsetNorm, layout_norm, extent_norm);
        cutlass::TensorView<ElementSum, LayoutNorm> view_Sum_Ref(
            offsetSum, layout_norm, extent_norm);

        // Reference GEMM-I.

        float alphaMma0 = usePreScaling ? alpha0 : 1.0f;
        float postScaling = usePreScaling ? 1.0f : alpha0;

        cutlass::DeviceAllocation<ElementP> block_Ref_P(
            layout_P.capacity(extent_P));
        cutlass::TensorView<ElementP, LayoutP> view_P(block_Ref_P.get(),
                                                      layout_P, extent_P);

        cutlass::reference::device::GemmComplex<
            ElementQ, LayoutQ, ElementK, LayoutK, ElementP, LayoutP,
            ElementAccumulator, ElementCompute>(
            problem0, ElementAccumulator(alphaMma0), view_Q,
            cutlass::ComplexTransform::kNone, view_K,
            cutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_P,
            view_P, ElementAccumulator(0));

        // Compute softmax for P. We need to explicitly compute softmax
        // over P because softmax is fused to the second GEMM in the
        // profiled implementation.
        std::vector<ElementP> matrix_Ref(layout_P.capacity(extent_P));
        cutlass::device_memory::copy_to_host(
            matrix_Ref.data(), block_Ref_P.get(), matrix_Ref.size());
        cutlass::TensorView<ElementP, LayoutP> view_Ref_host(
            matrix_Ref.data(), layout_P, extent_P);

        std::vector<ElementS> matrix_Ref_S(layout_P.capacity(extent_P));
        for (int i = 0; i < matrix_Ref.size(); ++i) {
          matrix_Ref_S.at(i) = ElementS(matrix_Ref.at(i));
        }
        // Copy to S (for debugging).
        cutlass::device_memory::copy_to_device(offsetS, matrix_Ref_S.data(),
                                               matrix_Ref_S.size());

        int n_dim_row = problem0.n();

        // Compute softmax for reference matrix
        if (usePow2) {
          for (int m = 0; m < problem0.m(); m++) {
            for (int n = 0; n < n_dim_row; n++) {
              view_Ref_host.ref().at({m, n}) =
                  kLog2e * postScaling * view_Ref_host.ref().at({m, n});
            }
            ElementSoftmaxCompute max =
                ElementSoftmaxCompute(view_Ref_host.ref().at({m, uint64_t(0)}));
            for (int n = 1; n < n_dim_row; n++) {
              max = std::max(
                  max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
            }

            view_Norm_Ref.at({m, uint64_t(0)}) = ElementNorm(max);

            ElementSoftmaxCompute sum = ElementSoftmaxCompute();
            for (int n = 0; n < n_dim_row; n++) {
              sum += std::exp2f(
                  ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max);
            }
            ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

            view_Sum_Ref.at({m, uint64_t(0)}) = ElementSum(inv_sum);

            for (int n = 0; n < n_dim_row; n++) {
              view_Ref_host.ref().at({m, n}) =
                  ElementP(std::exp2f(ElementSoftmaxCompute(
                                          view_Ref_host.ref().at({m, n})) -
                                      max) *
                           inv_sum);
            }
          }
        } else {
          for (int64_t m = 0; m < problem0.m(); m++) {
            ElementSoftmaxCompute max =
                ElementSoftmaxCompute(view_Ref_host.ref().at({m, uint64_t(0)}));
            for (int64_t n = 1; n < n_dim_row; n++) {
              max = std::max(
                  max, ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})));
            }

            view_Norm_Ref.at({m, uint64_t(0)}) = ElementNorm(max);

            ElementSoftmaxCompute sum = ElementSoftmaxCompute();
            for (int64_t n = 0; n < n_dim_row; n++) {
              sum += std::exp(
                  ElementSoftmaxCompute(view_Ref_host.ref().at({m, n})) - max);
            }
            ElementSoftmaxCompute inv_sum = ElementSoftmaxCompute(1.0f / sum);

            view_Sum_Ref.at({m, uint64_t(0)}) = ElementSum(inv_sum);

            for (int64_t n = 0; n < n_dim_row; n++) {
              view_Ref_host.ref().at({m, n}) =
                  ElementP(std::exp(ElementSoftmaxCompute(
                                        view_Ref_host.ref().at({m, n})) -
                                    max) *
                           inv_sum);
            }
          }
        }
        cutlass::device_memory::copy_to_device(
            block_Ref_P.get(), matrix_Ref.data(), matrix_Ref.size());

        // Reference GEMM-II.
        using Gemm2Type = cutlass::half_t;
        cutlass::reference::device::GemmComplex<
            ElementP, LayoutP, ElementV, LayoutV, ElementO, LayoutO,
            ElementAccumulator, ElementCompute>(
            problem1, ElementAccumulator(alpha1), view_P,
            cutlass::ComplexTransform::kNone, view_V,
            cutlass::ComplexTransform::kNone, ElementAccumulator(beta), view_O,
            view_O, ElementAccumulator(0));
      }
    }
  }
};
