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
#include "xtd/math/cbrt.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 4;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::cbrt(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::cbrt(x); };

TEST_CASE("xtd::cbrt", "[cbrt][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::cbrt(float)") {
    test_a<float, float, xtd::cbrt, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::cbrt(double)") {
    test_a<double, double, xtd::cbrt, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::cbrt(int)") {
    test_a<double, int, xtd::cbrt, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::cbrtf(float)") {
    test_f<float, float, xtd::cbrtf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::cbrtf(double)") {
    test_f<float, double, xtd::cbrtf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::cbrtf(int)") {
    test_f<float, int, xtd::cbrtf, ref_functionf>(values, ulps_single);
  }
}
