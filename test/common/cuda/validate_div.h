/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

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

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType)>
  __global__ static void kernel_div(InputType const* buf_x,
                                    InputType const* buf_y,
                                    ResultType* result_d,
                                    unsigned int size) {
    const unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      result_d[t] = static_cast<ResultType>(XtdFunc(buf_x[t], buf_y[t]));
    }
  }

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
    out << '{' << val.value_.quot << ", " << val.value_.rem << '}';
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

    // Generate a low-discrepancy deterministic sequence over [0, size) x [0, size)
    std::vector<InputType> input_x(size);
    std::vector<InputType> input_y(size);

    for (unsigned int t = 0; t < size; ++t) {
      auto [i, j] = halton<2>(t, size);
      input_x[t] = values_h[i];
      input_y[t] = values_h[j];
    }

    InputType* buf_x;
    InputType* buf_y;
    CUDA_CHECK(cudaMallocAsync(&buf_x, size * sizeof(InputType), queue));
    CUDA_CHECK(cudaMallocAsync(&buf_y, size * sizeof(InputType), queue));

    CUDA_CHECK(cudaMemcpyAsync(buf_x, input_x.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(buf_y, input_y.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    // Allocate memory for the results and fill it with zeroes
    std::vector<ResultType> result_h(size, ResultType{0, 0});
    ResultType* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

    // Execute the xtd function on the CUDA device
    kernel_div<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(buf_x, buf_y, result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to the host and free the GPU memory
    CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaFreeAsync(buf_x, queue));
    CUDA_CHECK(cudaFreeAsync(buf_y, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      InputType x = input_x[t];
      InputType y = input_y[t];
      // Read the result of the xtd function
      ResultType result = result_h[t];
      // Compute the reference
      ResultType reference = RefFunc(x, y);
      // Log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // Compare the result with the reference
      compare(result.quot, reference.quot);
      compare(result.rem, reference.rem);
    }
  }

}  // namespace test::cuda
