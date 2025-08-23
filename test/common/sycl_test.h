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

// test headers
#include "compare.h"

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(InputType)>
inline void test(sycl::queue queue, std::vector<double> const& values) {
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
    ResultType reference = StdFunc(input_h[i]);
    compare(result_h[i], reference);
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), ResultType (*StdFunc)(float)>
inline void test_f(sycl::queue queue, std::vector<double> const& values) {
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
    ResultType reference = StdFunc(static_cast<float>(input_h[i]));
    compare(result_h[i], reference);
  }
}
