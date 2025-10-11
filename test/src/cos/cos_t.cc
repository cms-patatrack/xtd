/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/cos.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::cos(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::cos(x); };

TEST_CASE("xtd::cos", "[cos][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::cos(float)") {
    test_a<float, float, xtd::cos, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::cos(double)") {
    test_a<double, double, xtd::cos, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::cos(int)") {
    test_a<double, int, xtd::cos, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::cosf(float)") {
    test_f<float, float, xtd::cosf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::cosf(double)") {
    test_f<float, double, xtd::cosf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::cosf(int)") {
    test_f<float, int, xtd::cosf, ref_functionf>(values, ulps_single);
  }
}
