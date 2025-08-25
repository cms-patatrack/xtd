/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <vector>

// HIP headers
#include <hip/hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// test headers
#include "compare.h"
#include "hip_check.h"

static constexpr auto single_prec = 24;
static constexpr auto double_prec = 53;
using mpfr_single = mpfr::real<single_prec, MPFR_RNDN>;
using mpfr_double = mpfr::real<double_prec, MPFR_RNDN>;

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType)>
__global__ static void kernel(InputType const* input, ResultType* result, int size) {
  const int thread = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (int i = thread; i < size; i += stride) {
    result[i] = static_cast<ResultType>(XtdFunc(input[i]));
  }
}

template <
    typename ResultType,
    typename InputType,
    ResultType (*XtdFunc)(InputType),
    typename std::enable_if<mpfr::type_traits<mpfr_double, mpfr_double, true>::enable_math_funcs,
                            const mpfr_double>::type (*RefFunc)(const mpfr_double&)>
inline void test(hipStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  HIP_CHECK(hipMallocAsync(&input_d, size * sizeof(InputType), queue));
  HIP_CHECK(hipMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), hipMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d;
  HIP_CHECK(hipMallocAsync(&result_d, size * sizeof(ResultType), queue));
  HIP_CHECK(hipMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size);
  HIP_CHECK(hipGetLastError());

  // copy the results back to the host and free the GPU memory
  HIP_CHECK(hipMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), hipMemcpyDeviceToHost, queue));
  HIP_CHECK(hipFreeAsync(input_d, queue));
  HIP_CHECK(hipFreeAsync(result_d, queue));
  HIP_CHECK(hipStreamSynchronize(queue));

  // compare the xtd results with std reference results
  for (int i = 0; i < size; ++i) {
    double input = input_h[i];
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    compare(result_h[i], reference, ulps);
  }
}

template <
    typename ResultType,
    typename InputType,
    ResultType (*XtdFunc)(InputType),
    typename std::enable_if<mpfr::type_traits<mpfr_single, mpfr_single, true>::enable_math_funcs,
                            const mpfr_single>::type (*RefFunc)(const mpfr_single&)>
inline void test_f(hipStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  HIP_CHECK(hipMallocAsync(&input_d, size * sizeof(InputType), queue));
  HIP_CHECK(hipMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), hipMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d;
  HIP_CHECK(hipMallocAsync(&result_d, size * sizeof(ResultType), queue));
  HIP_CHECK(hipMemsetAsync(result_d, 0x00, size * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size);
  HIP_CHECK(hipGetLastError());

  // copy the results back to the host and free the GPU memory
  HIP_CHECK(hipMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), hipMemcpyDeviceToHost, queue));
  HIP_CHECK(hipFreeAsync(input_d, queue));
  HIP_CHECK(hipFreeAsync(result_d, queue));
  HIP_CHECK(hipStreamSynchronize(queue));

  // compare the xtd results with std reference results
  for (int i = 0; i < size; ++i) {
    float input = static_cast<float>(input_h[i]);
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input)).conv(reference);
    compare(result_h[i], reference, ulps);
  }
}
