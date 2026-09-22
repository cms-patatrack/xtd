/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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

  template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType, InputType)>
  __global__ static void kernel_clamp(InputType const* buf_x,
                                      InputType const* buf_y,
                                      InputType const* buf_z,
                                      ResultType* result_d,
                                      unsigned int size) {
    const unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int t = thread; t < size; t += stride) {
      InputType x = buf_x[t];
      InputType y = xtd::min(buf_y[t], buf_z[t]);
      InputType z = xtd::max(buf_y[t], buf_z[t]);
      result_d[t] = static_cast<ResultType>(XtdFunc(x, y, z));
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

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate_clamp(const Device& device, int ulps = 0) {
    cudaStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();

    // Generate a low-discrepancy deterministic sequence over [0, size) x [0, size) x [0, size)
    std::vector<InputType> input_x(size);
    std::vector<InputType> input_y(size);
    std::vector<InputType> input_z(size);

    for (unsigned int t = 0; t < size; ++t) {
      auto [i, j, k] = halton<3>(t, size);
      input_x[t] = values_h[i];
      input_y[t] = values_h[j];
      input_z[t] = values_h[k];
    }

    InputType* buf_x;
    InputType* buf_y;
    InputType* buf_z;
    CUDA_CHECK(cudaMallocAsync(&buf_x, size * sizeof(InputType), queue));
    CUDA_CHECK(cudaMallocAsync(&buf_y, size * sizeof(InputType), queue));
    CUDA_CHECK(cudaMallocAsync(&buf_z, size * sizeof(InputType), queue));

    CUDA_CHECK(cudaMemcpyAsync(buf_x, input_x.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(buf_y, input_y.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(buf_z, input_z.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    // Allocate memory for the results and fill it with zeroes
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

    // Execute the xtd function on the GPU
    kernel_clamp<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(buf_x, buf_y, buf_z, result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to the host and free the GPU memory
    CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaFreeAsync(buf_x, queue));
    CUDA_CHECK(cudaFreeAsync(buf_y, queue));
    CUDA_CHECK(cudaFreeAsync(buf_z, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      InputType x = input_x[t];
      InputType y = std::min(input_y[t], input_z[t]);
      InputType z = std::max(input_y[t], input_z[t]);
      // Read the result of the xtd function
      ResultType result = result_h[t];
      // Compute the reference
      ResultType reference = RefFunc(x, y, z);
      // Log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // Compare the result with the reference
      compare(result, reference, ulps);
    }
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate_clamp(const Device& device) {
    cudaStream_t queue = device.queue();
    const Inputs& input = inputs(device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();

    // Generate a low-discrepancy deterministic sequence over [0, size) x [0, size) x [0, size)
    std::vector<InputType> input_x(size);
    std::vector<InputType> input_y(size);
    std::vector<InputType> input_z(size);

    for (unsigned int t = 0; t < size; ++t) {
      auto [i, j, k] = halton<3>(t, size);
      input_x[t] = values_h[i];
      input_y[t] = values_h[j];
      input_z[t] = values_h[k];
    }

    InputType* buf_x;
    InputType* buf_y;
    InputType* buf_z;
    CUDA_CHECK(cudaMallocAsync(&buf_x, size * sizeof(InputType), queue));
    CUDA_CHECK(cudaMallocAsync(&buf_y, size * sizeof(InputType), queue));
    CUDA_CHECK(cudaMallocAsync(&buf_z, size * sizeof(InputType), queue));

    CUDA_CHECK(cudaMemcpyAsync(buf_x, input_x.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(buf_y, input_y.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaMemcpyAsync(buf_z, input_z.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    // Allocate memory for the results and fill it with zeroes
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d;
    CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
    CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

    // Execute the xtd function on the CUDA device
    kernel_clamp<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(buf_x, buf_y, buf_z, result_d, size);
    CUDA_CHECK(cudaGetLastError());

    // Copy the results back to the host and free the device memory
    CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
    CUDA_CHECK(cudaFreeAsync(result_d, queue));
    CUDA_CHECK(cudaFreeAsync(buf_x, queue));
    CUDA_CHECK(cudaFreeAsync(buf_y, queue));
    CUDA_CHECK(cudaFreeAsync(buf_z, queue));
    CUDA_CHECK(cudaStreamSynchronize(queue));

    for (unsigned int t = 0; t < size; ++t) {
      // Generate a low-discrepancy deterministic sequence over [0, size) x [0, size) x [0, size)
      auto [i, j, k] = halton<3>(t, size);
      InputType x = values_h[i];
      InputType y = std::min(values_h[j], values_h[k]);
      InputType z = std::max(values_h[j], values_h[k]);
      // Read the result of the xtd function
      ResultType result = result_h[t];
      // Compute the reference
      ResultType reference = RefFunc(x, y, z);
      // Log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // Compare the result with the reference
      compare(result, reference);
    }
  }

}  // namespace test::cuda
