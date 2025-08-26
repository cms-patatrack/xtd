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

// SYCL headers
#include <sycl/sycl.hpp>

// mpfr::real headers
#include <real.hpp>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// test headers
#include "compare.h"

static constexpr auto single_prec = 24;
static constexpr auto double_prec = 53;
using mpfr_single = mpfr::real<single_prec, MPFR_RNDN>;
using mpfr_double = mpfr::real<double_prec, MPFR_RNDN>;

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_2(sycl::queue queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d = sycl::malloc_device<InputType>(size, queue);
  queue.copy(input_h.data(), input_d, size);

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d = sycl::malloc_device<ResultType>(outs, queue);
  queue.fill(result_d, ResultType{0}, outs);

  // execute the xtd function on the GPU
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(size * size / step), [=](sycl::id<1> k) {
      int i = k * step / size;
      int j = k * step % size;
      result_d[k] = static_cast<ResultType>(XtdFunc(input_d[i], input_d[j]));
    });
  });

  // copy the results back to the host and free the GPU memory
  queue.copy(result_d, result_h.data(), outs);
  queue.wait();
  sycl::free(input_d, queue);
  sycl::free(result_d, queue);

  // compare the xtd results with the reference results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    double input_y = input_h[i];
    double input_x = input_h[j];
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
inline void test_2f(sycl::queue queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  int outs = size * size / step + 1;

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d = sycl::malloc_device<InputType>(size, queue);
  queue.copy(input_h.data(), input_d, size);

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(outs, 0);
  ResultType* result_d = sycl::malloc_device<ResultType>(outs, queue);
  queue.fill(result_d, ResultType{0}, outs);

  // execute the xtd function on the GPU
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(size * size / step), [=](sycl::id<1> k) {
      int i = k * step / size;
      int j = k * step % size;
      result_d[k] = static_cast<ResultType>(XtdFunc(input_d[i], input_d[j]));
    });
  });

  // copy the results back to the host and free the GPU memory
  queue.copy(result_d, result_h.data(), outs);
  queue.wait();
  sycl::free(input_d, queue);
  sycl::free(result_d, queue);

  // compare the xtd results with the reference results
  for (int k = 0; k < size * size / step; ++k) {
    int i = k * step / size;
    int j = k * step % size;
    float input_y = static_cast<float>(input_h[i]);
    float input_x = static_cast<float>(input_h[j]);
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input_y), static_cast<mpfr_single>(input_x)).conv(reference);
    compare(result_h[k], reference, ulps);
  }
}
