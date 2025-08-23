/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <vector>

// mpfr::real headers
#include <real.hpp>

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
inline void test(std::vector<double> const& values, int ulps = 0) {
  for (double value : values) {
    // convert the input data to the type to be tested
    InputType input = static_cast<InputType>(value);
    // execute the xtd function
    ResultType result = XtdFunc(input);
    // compare the result with the std reference
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_double>(input)).conv(reference);
    compare(result, reference, ulps);
  }
}

template <
    typename ResultType,
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
    // compare the result with the std reference
    INFO(input);
    ResultType reference;
    RefFunc(static_cast<mpfr_single>(input)).conv(reference);
    compare(result, reference, ulps);
  }
}
