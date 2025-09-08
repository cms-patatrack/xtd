/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <type_traits>
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

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_double, mpfr_double, true>::enable_math_funcs,
                                  const mpfr_double>::type (*RefFunc)(const mpfr_double&)>
inline void test(sycl::queue queue, std::vector<double> const& values, int ulps = 0) try {
  if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
    if (not queue.get_device().has(sycl::aspect::fp64)) {
      INFO("The device does not support double precision floating point operations, the test will be skipped.");
      return;
    }
  }

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

  // compare the xtd results with the erence results
  for (int i = 0; i < size; ++i) {
    double input = input_h[i];
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    INFO(std::fixed << "input " << std::setprecision(std::numeric_limits<double>::max_digits10) << input
                    << ", xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception:\n"
            << e.what() << "\ncaught while running on platform "
            << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << ", device "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_single, mpfr_single, true>::enable_math_funcs,
                                  const mpfr_single>::type (*RefFunc)(const mpfr_single&)>
inline void test_f(sycl::queue queue, std::vector<double> const& values, int ulps = 0) try {
  if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
    if (not queue.get_device().has(sycl::aspect::fp64)) {
      INFO("The device does not support double precision floating point operations, the test will be skipped.");
      return;
    }
  }

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

  // compare the xtd results with the reference results
  for (int i = 0; i < size; ++i) {
    float input = static_cast<float>(input_h[i]);
    ResultType result = result_h[i];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input)).conv(reference);
    INFO(std::fixed << "input " << std::setprecision(std::numeric_limits<float>::max_digits10) << input
                    << ", xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception:\n"
            << e.what() << "\ncaught while running on platform "
            << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << ", device "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::exit(EXIT_FAILURE);
}

template <std::integral Type, Type (*XtdFunc)(Type), Type (*RefFunc)(Type)>
inline void test_i(sycl::queue queue, std::vector<double> const& values) try {
  int size = values.size();

  // convert the input data to the type to be tested and copy them to the GPU
  std::vector<Type> input_h(values.begin(), values.end());
  Type* input_d = sycl::malloc_device<Type>(size, queue);
  queue.copy(input_h.data(), input_d, size);

  // allocate memory for the results and fill it with zeroes
  std::vector<Type> result_h(size, 0);
  Type* result_d = sycl::malloc_device<Type>(size, queue);
  queue.fill(result_d, Type{0}, size);

  // execute the xtd function on the GPU
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> i) { result_d[i] = static_cast<Type>(XtdFunc(input_d[i])); });
  });

  // copy the results back to the host and free the GPU memory
  queue.copy(result_d, result_h.data(), size);
  queue.wait();
  sycl::free(input_d, queue);
  sycl::free(result_d, queue);

  // compare the xtd results with the reference results
  for (int i = 0; i < size; ++i) {
    Type input = input_h[i];
    Type result = result_h[i];
    Type reference = RefFunc(input);
    INFO("input " << input << ", xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception:\n"
            << e.what() << "\ncaught while running on platform "
            << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << ", device "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_2(sycl::queue queue, std::vector<double> const& values, int ulps = 0) try {
  if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
    if (not queue.get_device().has(sycl::aspect::fp64)) {
      INFO("The device does not support double precision floating point operations, the test will be skipped.");
      return;
    }
  }

  int size = values.size();
  int step = (size / 2) * 2 - 1;
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
    InputType input_y = input_h[i];
    InputType input_x = input_h[j];
    ResultType result = result_h[k];
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_y), static_cast<mpfr_double>(input_x)).conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input_y << ", "
                    << input_x << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception:\n"
            << e.what() << "\ncaught while running on platform "
            << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << ", device "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::exit(EXIT_FAILURE);
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single)>
inline void test_2f(sycl::queue queue, std::vector<double> const& values, int ulps = 0) try {
  if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
    if (not queue.get_device().has(sycl::aspect::fp64)) {
      INFO("The device does not support double precision floating point operations, the test will be skipped.");
      return;
    }
  }

  int size = values.size();
  int step = (size / 2) * 2 - 1;
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
    InputType input_y = static_cast<InputType>(input_h[i]);
    InputType input_x = static_cast<InputType>(input_h[j]);
    ResultType result = result_h[k];
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input_y)),
            static_cast<mpfr_single>(static_cast<float>(input_x)))
        .conv(reference);
    INFO(std::fixed << "inputs (" << std::setprecision(std::numeric_limits<float>::max_digits10) << input_y << ", "
                    << input_x << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10)
                    << result << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
} catch (sycl::exception const& e) {
  std::cerr << "SYCL exception:\n"
            << e.what() << "\ncaught while running on platform "
            << queue.get_device().get_platform().get_info<sycl::info::platform::name>() << ", device "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  std::exit(EXIT_FAILURE);
}
