/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <vector>
#include <type_traits>

// CUDA headers
#include <cuda_runtime.h>

// xtd headers
#include <xtd/algorithm.h>

// test headers
#include "common/compare.h"
#include "common/cuda/cuda_check.h"
#include "common/cuda/device.h"
#include "common/cuda/inputs.h"
#include "common/halton.h"
#include "common/mpfr.h"

namespace test::cuda {

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType)>
  __global__ static void kernel_div(InputType const* input, ResultType* result, unsigned int size) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = input[i];
      InputType y = input[j];
      result[t] = XtdFunc(x, y);
    }
  }

  template <typename T>
  concept DivResultType = requires(T t) {
    // C struct
    requires std::is_standard_layout_v<T>;

    // quot data member
    t.quot;
    requires std::integral<decltype(T::quot)>;

    // rem data member
    t.rem;
    requires std::integral<decltype(T::rem)>;

    // members type
    requires std::same_as<decltype(T::quot), decltype(T::rem)>;
  };

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <DivResultType T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << '[' << val.value_.quot << ", " << val.value_.rem << ']';
    return out;
  }

  template <DivResultType ResultType,
            std::integral InputType,
            ResultType (*XtdFunc)(InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType)>
  inline void validate_div(const Device& device) {
    cudaStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // allocate memory for the results and fill it with zeroes
    std::vector<ResultType> result_h(size);
    ResultType* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

    // execute the xtd function on the GPU
    kernel_div<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values_d.data(), result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // copy the results back to the host and free the GPU memory
    CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", "
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result.quot, reference.quot);
      compare(result.rem, reference.rem);
    }
  }

}  // namespace test::cuda
