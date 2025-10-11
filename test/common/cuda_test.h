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
#include "halton.h"

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

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), mpfr_double (*RefFunc)(mpfr_double)>
inline void test_a(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    // convert the input data to the type to be tested
    InputType input = input_h[i];
    // compare the xtd results with reference results
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    INFO(std::fixed << "input (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), mpfr_single (*RefFunc)(mpfr_single)>
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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(input_h[i]);
    // compare the xtd results with reference results
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input))).conv(reference);
    INFO(std::fixed << "input (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
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
  kernel<Type, Type, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(Type), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (int i = 0; i < size; ++i) {
    // convert the input data to the type to be tested
    Type input = input_h[i];
    // compare the xtd results with reference results
    Type result = result_h[i];
    Type reference = RefFunc(input);
    INFO("input (" << input << "), xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType, InputType)>
__global__ static void kernel(InputType const* input, ResultType* result, unsigned int size) {
  const int thread = blockDim.x * blockIdx.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  for (unsigned int t = thread; t < size; t += stride) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j] = halton<2>(t, size);
    result[t] = static_cast<ResultType>(XtdFunc(input[i], input[j]));
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_aa(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();

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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    ResultType result = result_h[t];
    // compare the xtd results with reference results
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_x), static_cast<mpfr_double>(input_y)).conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_x << ", "
                    << input_y << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single)>
inline void test_ff(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();

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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    // compare the xtd results with reference results
    ResultType result = result_h[t];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input_x)),
            static_cast<mpfr_single>(static_cast<float>(input_y)))
        .conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_x << ", "
                    << input_y << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <std::integral Type, Type (*XtdFunc)(Type, Type), Type (*RefFunc)(Type, Type)>
inline void test_ii(cudaStream_t queue, std::vector<double> const& values) {
  unsigned int size = values.size();

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
  kernel<Type, Type, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(Type), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    Type input_x = static_cast<Type>(values[i]);
    Type input_y = static_cast<Type>(values[j]);
    // compare the xtd results with reference results
    Type result = result_h[t];
    Type reference = RefFunc(input_x, input_y);
    INFO("inputs (" << input_x << ", " << input_y << "), xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double, mpfr_double)>
inline void test_aaa(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();

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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with erence results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    InputType input_z = static_cast<InputType>(values[k]);
    // compare the xtd results with reference results
    ResultType result = result_h[t];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_x), static_cast<mpfr_double>(input_y), static_cast<mpfr_double>(input_z))
        .conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_x << ", "
                    << input_y << ", " << input_z << "), xtd result "
                    << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result << " vs " << reference
                    << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single, mpfr_single)>
inline void test_fff(cudaStream_t queue, std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();

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
  kernel<ResultType, InputType, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(ResultType), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    InputType input_z = static_cast<InputType>(values[k]);
    // compare the xtd results with reference results
    ResultType result = result_h[t];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input_x)),
            static_cast<mpfr_single>(static_cast<float>(input_y)),
            static_cast<mpfr_single>(static_cast<float>(input_z)))
        .conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_x << ", "
                    << input_y << ", " << input_z << "), xtd result "
                    << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result << " vs " << reference
                    << '\n')
    compare(result, reference, ulps);
  }
}

template <std::integral Type, Type (*XtdFunc)(Type, Type, Type), Type (*RefFunc)(Type, Type, Type)>
inline void test_iii(cudaStream_t queue, std::vector<double> const& values) {
  unsigned int size = values.size();

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
  kernel<Type, Type, XtdFunc><<<8, 64, 0, queue>>>(input_d, result_d, size);
  CUDA_CHECK(cudaGetLastError());

  // copy the results back to the host and free the GPU memory
  CUDA_CHECK(cudaMemcpyAsync(result_h.data(), result_d, size * sizeof(Type), cudaMemcpyDeviceToHost, queue));
  CUDA_CHECK(cudaFreeAsync(input_d, queue));
  CUDA_CHECK(cudaFreeAsync(result_d, queue));
  CUDA_CHECK(cudaStreamSynchronize(queue));

  // compare the xtd results with the reference results
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)*[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    Type input_x = static_cast<Type>(values[i]);
    Type input_y = static_cast<Type>(values[j]);
    Type input_z = static_cast<Type>(values[k]);
    // compare the xtd results with reference results
    Type result = result_h[t];
    Type reference = RefFunc(input_x, input_y, input_z);
    INFO("inputs (" << input_x << ", " << input_y << ", " << input_z << "), xtd result " << result << " vs "
                    << reference << '\n')
    compare(result, reference);
  }
}
