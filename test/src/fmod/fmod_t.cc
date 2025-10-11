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
#include "xtd/math/fmod.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::fmod(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::fmod(x, y); };

TEST_CASE("xtd::fmod", "[fmod][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::fmod(float, float)") {
    test_aa<float, float, xtd::fmod, ref_function>(values, ulps_single);
  }

  SECTION("double xtd::fmod(double, double)") {
    test_aa<double, double, xtd::fmod, ref_function>(values, ulps_double);
  }

  SECTION("double xtd::fmod(int, int)") {
    test_aa<double, int, xtd::fmod, ref_function>(values, ulps_double);
  }

  SECTION("float xtd::fmodf(float, float)") {
    test_ff<float, float, xtd::fmodf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::fmodf(double, double)") {
    test_ff<float, double, xtd::fmodf, ref_functionf>(values, ulps_single);
  }

  SECTION("float xtd::fmodf(int, int)") {
    test_ff<float, int, xtd::fmodf, ref_functionf>(values, ulps_single);
  }
}
