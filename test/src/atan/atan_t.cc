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
#include "xtd/math/atan.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::atan(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::atan(x); };

TEST_CASE("xtd::atan", "[atan][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atan(float)") {
    test_a<float, float, xtd::atan, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::atan(double)") {
    test_a<double, double, xtd::atan, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::atan(int)") {
    test_a<double, int, xtd::atan, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::atanf(float)") {
    test_f<float, float, xtd::atanf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atanf(double)") {
    test_f<float, double, xtd::atanf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atanf(int)") {
    test_f<float, int, xtd::atanf, ref_functionf>(values, ulps_single);
  }
}
