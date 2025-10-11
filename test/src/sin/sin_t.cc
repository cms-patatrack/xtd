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
#include "xtd/math/sin.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::sin(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::sin(x); };

TEST_CASE("xtd::sin", "[sin][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::sin(float)") {
    test_a<float, float, xtd::sin, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::sin(double)") {
    test_a<double, double, xtd::sin, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::sin(int)") {
    test_a<double, int, xtd::sin, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::sinf(float)") {
    test_f<float, float, xtd::sinf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::sinf(double)") {
    test_f<float, double, xtd::sinf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::sinf(int)") {
    test_f<float, int, xtd::sinf, ref_functionf>(values, ulps_single);
  }
}
