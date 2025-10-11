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
#include "xtd/math/atanh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::atanh(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::atanh(x); };

TEST_CASE("xtd::atanh", "[atanh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atanh(float)") {
    test_a<float, float, xtd::atanh, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::atanh(double)") {
    test_a<double, double, xtd::atanh, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::atanh(int)") {
    test_a<double, int, xtd::atanh, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::atanhf(float)") {
    test_f<float, float, xtd::atanhf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atanhf(double)") {
    test_f<float, double, xtd::atanhf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::atanhf(int)") {
    test_f<float, int, xtd::atanhf, ref_functionf>(values, ulps_single);
  }
}
