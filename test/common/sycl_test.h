/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <vector>

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

template <
    typename ResultType,
    typename InputType,
    ResultType (*XtdFunc)(InputType),
    typename std::enable_if<mpfr::type_traits<mpfr_double, mpfr_double, true>::enable_math_funcs,
                            const mpfr_double>::type (*RefFunc)(const mpfr_double&)>
inline void test(sycl::queue queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d = sycl::malloc_device<InputType>(size, queue);
  queue.copy(input_h.data(), input_d, size);

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d = sycl::malloc_device<ResultType>(size, queue);
  queue.fill(result_d, ResultType{0}, size);

  // execute the xtd function on the GPU
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> i) { result_d[i] = static_cast<ResultType>(XtdFunc(input_d[i])); });
  });

  // copy the results back to the host and free the GPU memory
  queue.copy(result_d, result_h.data(), size);
  queue.wait();
  sycl::free(input_d, queue);
  sycl::free(result_d, queue);

  // compare the xtd results with the std reference results
  for (int i = 0; i < size; ++i) {
    INFO(input_h[i]);
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_h[i])).conv(reference);
    compare<float>(result_h[i], reference, ulps);
  }
}

template <
    typename ResultType,
    typename InputType,
    ResultType (*XtdFunc)(InputType),
    typename std::enable_if<mpfr::type_traits<mpfr_single, mpfr_single, true>::enable_math_funcs,
                            const mpfr_single>::type (*RefFunc)(const mpfr_single&)>
inline void test_f(sycl::queue queue, std::vector<double> const& values, int ulps = 0) {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<InputType> input_h(values.begin(), values.end());
  InputType* input_d = sycl::malloc_device<InputType>(size, queue);
  queue.copy(input_h.data(), input_d, size);

  // allocate memory for the results and fill it with zeroes
  std::vector<ResultType> result_h(size, 0);
  ResultType* result_d = sycl::malloc_device<ResultType>(size, queue);
  queue.fill(result_d, ResultType{0}, size);

  // execute the xtd function on the GPU
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> i) { result_d[i] = static_cast<ResultType>(XtdFunc(input_d[i])); });
  });

  // copy the results back to the host and free the GPU memory
  queue.copy(result_d, result_h.data(), size);
  queue.wait();
  sycl::free(input_d, queue);
  sycl::free(result_d, queue);

  // compare the xtd results with the std reference results
  for (int i = 0; i < size; ++i) {
    INFO(input_h[i]);
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input_h[i])).conv(reference);
    compare<float>(result_h[i], reference, ulps);
  }
}
