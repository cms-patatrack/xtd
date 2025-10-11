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
#include "xtd/math/tan.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::tan(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::tan(x); };

TEST_CASE("xtd::tan", "[tan][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::tan(float)") {
    test_a<float, float, xtd::tan, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::tan(double)") {
    test_a<double, double, xtd::tan, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::tan(int)") {
    test_a<double, int, xtd::tan, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::tanf(float)") {
    test_f<float, float, xtd::tanf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::tanf(double)") {
    test_f<float, double, xtd::tanf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::tanf(int)") {
    test_f<float, int, xtd::tanf, ref_functionf>(values, ulps_single);
  }
}
