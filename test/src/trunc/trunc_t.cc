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
#include "xtd/math/trunc.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::trunc(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::trunc(x); };

TEST_CASE("xtd::trunc", "[trunc][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::trunc(float)") {
    test_a<float, float, xtd::trunc, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::trunc(double)") {
    test_a<double, double, xtd::trunc, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::trunc(int)") {
    test_a<double, int, xtd::trunc, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::truncf(float)") {
    test_f<float, float, xtd::truncf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::truncf(double)") {
    test_f<float, double, xtd::truncf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::truncf(int)") {
    test_f<float, int, xtd::truncf, ref_functionf>(values, ulps_single);
  }
}
