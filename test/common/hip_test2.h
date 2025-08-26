/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string>
#include <vector>
using namespace std::literals;

// HIP headers
#include <hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// test headers
#include "compare.h"
#include "hip_check.h"

static constexpr auto single_prec = 24;
static constexpr auto double_prec = 53;
using mpfr_single = mpfr::real<single_prec, MPFR_RNDN>;
using mpfr_double = mpfr::real<double_prec, MPFR_RNDN>;

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType)>
__global__ static void kernel(InputType const* input, ResultType* result, int size, int step) {
  const int thread = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int k = thread; k < size * size / step; k += stride) {
    int i = k * step / size;
    int j = k * step % size;
    result[k] = static_cast<ResultType>(XtdFunc(input[i], input[j]));
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_2(hipStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  HIP_CHECK(hipMallocAsync(&input_d, size * sizeof(InputType), queue));
  HIP_CHECK(hipMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), hipMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d;
  HIP_CHECK(hipMallocAsync(&result_d, outs * sizeof(ResultType), queue));
  HIP_CHECK(hipMemsetAsync(result_d, 0x00, outs * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size, step);
  HIP_CHECK(hipGetLastError());

  // copy the results back to the host and free the GPU memory
  HIP_CHECK(hipMemcpyAsync(result_h.data(), result_d, outs * sizeof(ResultType), hipMemcpyDeviceToHost, queue));
  HIP_CHECK(hipFreeAsync(input_d, queue));
  HIP_CHECK(hipFreeAsync(result_d, queue));
  HIP_CHECK(hipStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_y), static_cast<mpfr_double>(input_x)).conv(reference);
    compare(result_h[k], reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single)>
inline void test_2f(hipStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  HIP_CHECK(hipMallocAsync(&input_d, size * sizeof(InputType), queue));
  HIP_CHECK(hipMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), hipMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d;
  HIP_CHECK(hipMallocAsync(&result_d, outs * sizeof(ResultType), queue));
  HIP_CHECK(hipMemsetAsync(result_d, 0x00, outs * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size, step);
  HIP_CHECK(hipGetLastError());

  // copy the results back to the host and free the GPU memory
  HIP_CHECK(hipMemcpyAsync(result_h.data(), result_d, outs * sizeof(ResultType), hipMemcpyDeviceToHost, queue));
  HIP_CHECK(hipFreeAsync(input_d, queue));
  HIP_CHECK(hipFreeAsync(result_d, queue));
  HIP_CHECK(hipStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    float input_y = static_cast<float>(static_cast<InputType>(values[i]));
    float input_x = static_cast<float>(static_cast<InputType>(values[j]));
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input_y), static_cast<mpfr_single>(input_x)).conv(reference);
    compare(result_h[k], reference, ulps);
  }
}
