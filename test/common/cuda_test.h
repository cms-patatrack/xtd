/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <vector>

// CUDA headers
#include <cuda_runtime.h>

// test headers
#include "compare.h"
#include "cuda_check.h"

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType)>
__global__ static void kernel(InputType const* input, ResultType* result, int size) {
  const int thread = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = thread; i < size; i += stride) {
    result[i] = static_cast<ResultType>(XtdFunc(input[i]));
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(InputType)>
inline void test(cudaStream_t queue, std::vector<double> const& values) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  CUDA_CHECK(cudaMallocAsync(&input_d, size * sizeof(InputType), queue));
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d;
  CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
  CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with std reference results
  for (int i = 0; i < size; ++i) {
    ResultType reference = StdFunc(input_h[i]);
    compare(result_h[i], reference);
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(float)>
inline void test_f(cudaStream_t queue, std::vector<double> const& values) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  CUDA_CHECK(cudaMallocAsync(&input_d, size * sizeof(InputType), queue));
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d;
  CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(ResultType), queue));
  CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with std reference results
  for (int i = 0; i < size; ++i) {
    ResultType reference = StdFunc(static_cast<float>(input_h[i]));
    compare(result_h[i], reference);
  }
}
