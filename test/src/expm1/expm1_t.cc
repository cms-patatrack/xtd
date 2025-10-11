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
#include "xtd/math/expm1.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::expm1(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::expm1(x); };

TEST_CASE("xtd::expm1", "[expm1][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::expm1(float)") {
    test_a<float, float, xtd::expm1, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::expm1(double)") {
    test_a<double, double, xtd::expm1, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::expm1(int)") {
    test_a<double, int, xtd::expm1, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::expm1f(float)") {
    test_f<float, float, xtd::expm1f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::expm1f(double)") {
    test_f<float, double, xtd::expm1f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::expm1f(int)") {
    test_f<float, int, xtd::expm1f, ref_functionf>(values, ulps_single);
  }
}
