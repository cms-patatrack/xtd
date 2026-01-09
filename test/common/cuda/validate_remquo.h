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

  template <std::floating_point T>
  struct remquo_t {
    T rem;
    int quo;
  };

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType, int*)>
  __global__ static void kernel_remquo(InputType const* input, remquo_t<ResultType>* result, unsigned int size) {
    const int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = input[i];
      InputType y = input[j];
      result[t].rem = XtdFunc(x, y, &result[t].quo);
    }
  }

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << val.value_ << " ["
           << std::hexfloat << val.value_ << "]";
    out << buffer.str();
    return out;
  }

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<remquo_t<T>> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << "{" << val.value_.rem << " ["
           << std::hexfloat << val.value_.rem << "], " << val.value_.quo << "}";
    out << buffer.str();
    return out;
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, int*),
            ResultType (*RefFunc)(InputType, InputType, int*)>
  inline void validate_remquo(const Device& device, int ulps = 0) {
    cudaStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // allocate memory for the results and fill it with zeroes
    std::vector<remquo_t<ResultType>> result_h(size);
    remquo_t<ResultType>* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(remquo_t<ResultType>), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(remquo_t<ResultType>), queue));

    // execute the xtd function on the GPU
    kernel_remquo<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(values_d.data(), result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // copy the results back to the host and free the GPU memory
    CUDA_CHECK(
        cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(remquo_t<ResultType>), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      // read the result of the xtd function
      remquo_t<ResultType> result = result_h[t];
      // compute the reference
      remquo_t<ResultType> reference;
      reference.rem = RefFunc(x, y, &reference.quo);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", "
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result.rem, reference.rem, ulps);
      // the standard guarantees only the sign and the last three bits
      compare(result.quo % 8, reference.quo % 8);
    }
  }

}  // namespace test::cuda
