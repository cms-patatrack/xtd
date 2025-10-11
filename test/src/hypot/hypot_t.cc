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
#include "xtd/math/hypot.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::hypot(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::hypot(x, y); };

TEST_CASE("xtd::hypot", "[hypot][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::hypot(float, float)") {
    test_aa<float, float, xtd::hypot, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::hypot(double, double)") {
    test_aa<double, double, xtd::hypot, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::hypot(int, int)") {
    test_aa<double, int, xtd::hypot, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::hypotf(float, float)") {
    test_ff<float, float, xtd::hypotf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::hypotf(double, double)") {
    test_ff<float, double, xtd::hypotf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::hypotf(int, int)") {
    test_ff<float, int, xtd::hypotf, ref_functionf>(values, ulps_single);
  }
}
