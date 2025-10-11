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
#include "xtd/math/acosh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::acosh(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::acosh(x); };

TEST_CASE("xtd::acosh", "[acosh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::acosh(float)") {
    test_a<float, float, xtd::acosh, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::acosh(double)") {
    test_a<double, double, xtd::acosh, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::acosh(int)") {
    test_a<double, int, xtd::acosh, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::acoshf(float)") {
    test_f<float, float, xtd::acoshf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::acoshf(double)") {
    test_f<float, double, xtd::acoshf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::acoshf(int)") {
    test_f<float, int, xtd::acoshf, ref_functionf>(values, ulps_single);
  }
}
