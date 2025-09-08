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

// mpfr::real headers
#include <real.hpp>

// test headers
#include "compare.h"
#include "halton.h"

static constexpr auto single_prec = 24;
static constexpr auto double_prec = 53;
using mpfr_single = mpfr::real<single_prec, MPFR_RNDN>;
using mpfr_double = mpfr::real<double_prec, MPFR_RNDN>;

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_double, mpfr_double, true>::enable_math_funcs,
                                  const mpfr_double>::type (*RefFunc)(const mpfr_double&)>
inline void test(std::vector<double> const& values, int ulps = 0) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the reference
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType),
          typename std::enable_if<mpfr::type_traits<mpfr_single, mpfr_single, true>::enable_math_funcs,
                                  const mpfr_single>::type (*RefFunc)(const mpfr_single&)>
inline void test_f(std::vector<double> const& values, int ulps = 0) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the reference
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input))).conv(reference);
    compare(result, reference, ulps);
  }
}

template <std::integral Type, Type (*XtdFunc)(Type), Type (*RefFunc)(Type)>
inline void test_i(std::vector<double> const& values) {
  for (double value : values) {
    // convert the input data to the type to be tested
    Type input = static_cast<Type>(value);
    // execute the xtd function
    Type result = XtdFunc(input);
    // compare the xtd results with reference results
    INFO(input);
    Type reference = RefFunc(input);
    compare(result, reference);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_2(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_y, input_x);
    // compare the result with the reference
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
inline void test_2f(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_y, input_x);
    // compare the result with the reference
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

template <std::integral Type, Type (*XtdFunc)(Type), Type (*RefFunc)(Type, Type)>
inline void test_2i(std::vector<double> const& values) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    Type input_y = static_cast<Type>(values[i]);
    Type input_x = static_cast<Type>(values[j]);
    // execute the xtd function
    Type result = XtdFunc(input_y, input_x);
    // compare the xtd results with reference results
    Type reference = RefFunc(input_y, input_x);
    INFO("inputs (" << input_y << ", " << input_x << "), xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}
