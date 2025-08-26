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

// mpfr::real headers
#include <real.hpp>

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
inline void test_2(std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  for (unsigned int k = 0; k < size * size; k += step) {
    int i = k / size;
    int j = k % size;
    // convert the input data to the type to be tested
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_y, input_x);
    // compare the result with the reference
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input_y), static_cast<mpfr_double>(input_x)).conv(reference);
    compare(result, reference, ulps);
  }
}

template <typename ResultType,
          typename InputType,
          ResultType (*XtdFunc)(InputType, InputType),
          mpfr_single (*RefFunc)(mpfr_single, mpfr_single)>
inline void test_2f(std::vector<double> const& values, int ulps = 0) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  for (unsigned int k = 0; k < size * size; k += step) {
    int i = k / size;
    int j = k % size;
    // convert the input data to the type to be tested
    InputType input_y = static_cast<InputType>(values[i]);
    InputType input_x = static_cast<InputType>(values[j]);
    // execute the xtd function
    ResultType result = XtdFunc(input_y, input_x);
    // compare the result with the reference
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(static_cast<float>(input_y)),
            static_cast<mpfr_single>(static_cast<float>(input_x)))
        .conv(reference);
    compare(result, reference, ulps);
  }
}

template <std::integral Type, Type (*XtdFunc)(Type), Type (*RefFunc)(Type, Type)>
inline void test_2i(std::vector<double> const& values) {
  int size = values.size();
  int step = std::trunc(std::sqrt(size)) - 1;
  for (unsigned int k = 0; k < size * size; k += step) {
    int i = k / size;
    int j = k % size;
    // convert the input data to the type to be tested
    Type input_y = static_cast<Type>(values[i]);
    Type input_x = static_cast<Type>(values[j]);
    // execute the xtd function
    Type result = XtdFunc(input_y, input_x);
    // compare the xtd results with reference results
    INFO(std::to_string(input_y) + ", "s + std::to_string(input_x));
    Type reference = RefFunc(input_y, input_x);
    compare(result, reference);
  }
}
