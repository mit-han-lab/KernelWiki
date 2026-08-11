#pragma once

#include <iomanip>

template <typename T> struct TransposeParams {
  T *__restrict__ input;
  T *__restrict__ output;

  const int M;
  const int N;

  TransposeParams(T *__restrict__ input_, T *__restrict__ output_, int M_, int N_)
      : input(input_), output(output_), M(M_), N(N_) {}
};

// template <typename T> int benchmark(void (*transpose)(int M, int N, T* input, T* output), int M, int N, int iterations=10, bool verify=true) {
template <typename T, bool isTranspose = true, bool isFMA = false> int benchmark(void (*transpose)(TransposeParams<T> params), int M, int N, int iterations=10, bool verify=true) {
  using namespace cute;

  auto tensor_shape_S = make_shape(M, N);
  auto tensor_shape_D = (isTranspose) ? make_shape(N, M) : make_shape(M, N);

  // Allocate and initialize
  thrust::host_vector<T> h_S(size(tensor_shape_S));       // (M, N)
  thrust::host_vector<T> h_D(size(tensor_shape_D)); // (N, M)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<T>(i);

  thrust::device_vector<T> d_S = h_S;
  thrust::device_vector<T> d_D = h_D;

  TransposeParams<T> params(thrust::raw_pointer_cast(d_S.data()), thrust::raw_pointer_cast(d_D.data()), M, N);

  
  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    transpose(params);
    cudaError result = cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    if (result != cudaSuccess) {
      std::cerr << "CUDA Runtime error: " << cudaGetErrorString(result)
                << std::endl;
      return -1;
    }
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    int numThreads = 256;
    // int numThreads = 128;
    double M_ = double(M);
    double N_ = double(N);
    double bytes = !isFMA ? 2 * M_ * N_ * sizeof(T) : (M_ * N_ + M_ * numThreads) * sizeof(T);
    
    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << std::fixed << std::setprecision(2) 
              << 1e-6 * bytes / time_ms << " GB/s)"
              << std::endl;

    if(isFMA) {
      uint64_t flops = numThreads * 2 * uint64_t(M) * uint64_t(N);
      double AI = double(flops) / double(bytes);
      std::cout << "Arithmetic Intensity = " << AI << std::endl;
      std::cout << "TFLOPs/s = "
                << std::fixed << std::setprecision(2) 
                << (1e-9 * flops / time_ms) << std::endl;

    }
  }

  if(verify) {
    h_D = d_D;
  
    int bad = 0;
    if constexpr (isTranspose) {
      auto transpose_function = make_layout(tensor_shape_S, LayoutRight{});
      for (size_t i = 0; i < h_D.size(); ++i) 
        if (h_D[i] != h_S[transpose_function(i)])
          bad++;
    } else {
      for (size_t i = 0; i < h_D.size(); ++i) 
        if (h_D[i] != h_S[i])
          bad++;
    }
  
    if (bad > 0) {
      std::cout << "Validation failed. Correct values: " << h_D.size()-bad << ". Incorrect values: " << bad << std::endl;
    } else {
      std::cout << "Validation success." << std::endl;
    }
  }
  return 0;
}
