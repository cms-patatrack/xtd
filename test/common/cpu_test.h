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

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), mpfr_double (*RefFunc)(mpfr_double)>
inline void test_a(std::vector<double> const& values, int ulps = 0) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the reference
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(static_cast<double>(input))).conv(reference);
    INFO(std::fixed << "input (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
    compare(result, reference, ulps);
  }
}

template <typename ResultType, typename InputType, ResultType (*XtdFunc)(InputType), mpfr_single (*RefFunc)(mpfr_single)>
inline void test_f(std::vector<double> const& values, int ulps = 0) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the reference
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input))).conv(reference);
    INFO(std::fixed << "input (" << std::setprecision(std::numeric_limits<InputType>::max_digits10) << input
                    << "), xtd result " << std::setprecision(std::numeric_limits<ResultType>::max_digits10) << result
                    << " vs " << reference << '\n')
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
    Type reference = RefFunc(input);
    INFO("input (" << input << "), xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double)>
inline void test_aa(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_x, input_y);
    // compare the result with the reference
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
inline void test_ff(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_x, input_y);
    // compare the result with the reference
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
inline void test_ii(std::vector<double> const& values) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
    auto [i, j] = halton<2>(t, size);
    // convert the input data to the type to be tested
    Type input_x = static_cast<Type>(values[i]);
    Type input_y = static_cast<Type>(values[j]);
    // execute the xtd function
    Type result = XtdFunc(input_x, input_y);
    // compare the xtd results with reference results
    Type reference = RefFunc(input_x, input_y);
    INFO("inputs (" << input_x << ", " << input_y << "), xtd result " << result << " vs " << reference << '\n')
    compare(result, reference);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType, InputType),
          mpfr_double (*RefFunc)(mpfr_double, mpfr_double, mpfr_double)>
inline void test_aaa(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    InputType input_z = static_cast<InputType>(values[k]);
    // execute the xtd function
    ResultType result = XtdFunc(input_x, input_y, input_z);
    // compare the result with the reference
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(static_cast<double>(input_x)),
            static_cast<mpfr_double>(static_cast<double>(input_y)),
            static_cast<mpfr_double>(static_cast<double>(input_z)))
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
inline void test_fff(std::vector<double> const& values, int ulps = 0) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    InputType input_x = static_cast<InputType>(values[i]);
    InputType input_y = static_cast<InputType>(values[j]);
    InputType input_z = static_cast<InputType>(values[k]);
    // execute the xtd function
    ResultType result = XtdFunc(input_x, input_y, input_z);
    // compare the result with the reference
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
inline void test_iii(std::vector<double> const& values) {
  unsigned int size = values.size();
  for (unsigned int t = 0; t < size; ++t) {
    // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size)
    auto [i, j, k] = halton<3>(t, size);
    // convert the input data to the type to be tested
    Type input_x = static_cast<Type>(values[i]);
    Type input_y = static_cast<Type>(values[j]);
    Type input_z = static_cast<Type>(values[k]);
    // execute the xtd function
    Type result = XtdFunc(input_x, input_y, input_z);
    // compare the xtd results with reference results
    Type reference = RefFunc(input_x, input_y, input_z);
    INFO("inputs (" << input_x << ", " << input_y << ", " << input_z << "), xtd result " << result << " vs "
                    << reference << '\n')
    compare(result, reference);
  }
}
