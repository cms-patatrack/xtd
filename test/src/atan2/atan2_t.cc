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
#include "xtd/math/atan2.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::atan2(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::atan2(x, y); };

TEST_CASE("xtd::atan2", "[atan2][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atan2(float, float)") {
    test_aa<float, float, xtd::atan2, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::atan2(double, double)") {
    test_aa<double, double, xtd::atan2, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::atan2(int, int)") {
    test_aa<double, int, xtd::atan2, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::atan2f(float, float)") {
    test_ff<float, float, xtd::atan2f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atan2f(double, double)") {
    test_ff<float, double, xtd::atan2f, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atan2f(int, int)") {
    test_ff<float, int, xtd::atan2f, ref_functionf>(values, ulps_single);
  }
}
