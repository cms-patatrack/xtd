/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <limits>
#include <span>
#include <sstream>
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
  __global__ static void kernel_remquo(InputType const* buf_x,
                                       InputType const* buf_y,
                                       remquo_t<ResultType>* result_d,
                                       unsigned int size) {
    const unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      result_d[t].rem = static_cast<ResultType>(XtdFunc(buf_x[t], buf_y[t], &result_d[t].quo));
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
    std::vector<remquo_t<ResultType>> result_h(size, remquo_t<ResultType>{0., 0});
    remquo_t<ResultType>* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(remquo_t<ResultType>), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(remquo_t<ResultType>), queue));

    // Execute the xtd function on the CUDA device
    kernel_remquo<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(buf_x, buf_y, result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to the host and free the GPU memory
    CUDA_CHECK(
        cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(remquo_t<ResultType>), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaFreeAsync(buf_x, queue));
    CUDA_CHECK(cudaFreeAsync(buf_y, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      InputType x = input_x[t];
      InputType y = input_y[t];
      // Read the result of the xtd function
      remquo_t<ResultType> result = result_h[t];
      // Compute the reference
      remquo_t<ResultType> reference;
      reference.rem = RefFunc(x, y, &reference.quo);
      // Log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // Compare the result with the reference
      compare(result.rem, reference.rem, ulps);
      // The standard guarantees only the sign and the last three bits
      compare(result.quo % 8, reference.quo % 8);
    }
  }

}  // namespace test::cuda
