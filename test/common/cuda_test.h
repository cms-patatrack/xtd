/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
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

// mpfr::real headers
#include <real.hpp>

// test headers
#include "compare.h"
#include "cuda_check.h"

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

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_double, mpfr_double, true>::enable_math_funcs,
                                  const mpfr_double>::type (*RefFunc)(const mpfr_double&)>
inline void test(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
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

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    InputType input = input_h[i];
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    INFO(std::fixed << "input " << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << ", xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_single, mpfr_single, true>::enable_math_funcs,
                                  const mpfr_single>::type (*RefFunc)(const mpfr_single&)>
inline void test_f(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
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

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    InputType input = static_cast<InputType>(input_h[i]);
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input))).conv(reference);
    INFO(std::fixed << "input " << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << ", xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <std::integral Type, Type (*XtdFunc)(Type), Type (*RefFunc)(Type)>
inline void test_i(cudaStream_t queue, std::vector<double> const& values) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<Type> input_h(values.begin(), values.end());
  Type* input_d;
  CUDA_CHECK(cudaMallocAsync(&input_d, size * sizeof(Type), queue));
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), size * sizeof(Type), cudaMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<Type> result_h(size, 0);
  Type* result_d;
  CUDA_CHECK(cudaMallocAsync(&result_d, size * sizeof(Type), queue));
  CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, size * sizeof(Type), queue));

  // execute the xtd function on the GPU
  kernel<Type, Type, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(Type), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    Type input = input_h[i];
    Type result = result_h[i];
    Type reference = RefFunc(input);
    INFO("input " << input << ", xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}

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
inline void test_2(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  CUDA_CHECK(cudaMallocAsync(&input_d, size * sizeof(InputType), queue));
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d;
  CUDA_CHECK(cudaMallocAsync(&result_d, outs * sizeof(ResultType), queue));
  CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, outs * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size, step);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, outs * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    ResultType result = result_h[k];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_y), static_cast<mpfr_double>(input_x)).conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_y << ", "
                    << input_x << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single)>
inline void test_2f(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d;
  CUDA_CHECK(cudaMallocAsync(&input_d, size * sizeof(InputType), queue));
  CUDA_CHECK(cudaMemcpyAsync(input_d, input_h.data(), size * sizeof(InputType), cudaMemcpyHostToDevice, queue));

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d;
  CUDA_CHECK(cudaMallocAsync(&result_d, outs * sizeof(ResultType), queue));
  CUDA_CHECK(cudaMemsetAsync(result_d, 0x00, outs * sizeof(ResultType), queue));

  // execute the xtd function on the GPU
  kernel<ResultType, InputType, XtdFunc><<<8, 32, 0, queue>>>(input_d, result_d, size, step);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, outs * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    ResultType result = result_h[k];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input_y)),
            static_cast<mpfr_single>(static_cast<float>(input_x)))
        .conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_y << ", "
                    << input_x << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}
