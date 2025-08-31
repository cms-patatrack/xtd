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

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmax = [](mpfr_double y, mpfr_double x) { return mpfr::fmax(y, x); };
constexpr auto ref_fmaxf = [](mpfr_single y, mpfr_single x) { return mpfr::fmax(y, x); };

TEST_CASE("xtd::fmax", "[fmax][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::fmax(float, float)") {
    test_2<float, float, xtd::fmax, ref_fmax>(values, ulps_float);
  }

  SECTION("double xtd::fmax(double, double)") {
    test_2<double, double, xtd::fmax, ref_fmax>(values, ulps_double);
  }

  SECTION("double xtd::fmax(int, int)") {
    test_2<double, int, xtd::fmax, ref_fmax>(values, ulps_double);
  }

  SECTION("float xtd::fmaxf(float, float)") {
    test_2f<float, float, xtd::fmaxf, ref_fmaxf>(values, ulps_float);
  }

  SECTION("float xtd::fmaxf(double, double)") {
    test_2f<float, double, xtd::fmaxf, ref_fmaxf>(values, ulps_float);
  }

  SECTION("float xtd::fmaxf(int, int)") {
    test_2f<float, int, xtd::fmaxf, ref_fmaxf>(values, ulps_float);
  }
}
