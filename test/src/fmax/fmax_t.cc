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
#include "xtd/math/fmax.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::fmax(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::fmax(x, y); };

TEST_CASE("xtd::fmax", "[fmax][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::fmax(float, float)") {
    test_aa<float, float, xtd::fmax, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::fmax(double, double)") {
    test_aa<double, double, xtd::fmax, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::fmax(int, int)") {
    test_aa<double, int, xtd::fmax, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::fmaxf(float, float)") {
    test_ff<float, float, xtd::fmaxf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::fmaxf(double, double)") {
    test_ff<float, double, xtd::fmaxf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::fmaxf(int, int)") {
    test_ff<float, int, xtd::fmaxf, ref_functionf>(values, ulps_single);
  }
}
