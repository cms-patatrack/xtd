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

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

constexpr auto ref_atan2 = [](mpfr_double y, mpfr_double x) { return mpfr::atan2(y, x); };
constexpr auto ref_atan2f = [](mpfr_single y, mpfr_single x) { return mpfr::atan2(y, x); };

TEST_CASE("xtd::atan2", "[atan2][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atan2(float, float)") {
    test_2<float, float, xtd::atan2, ref_atan2>(values, ulps_float);
  }

  SECTION("double xtd::atan2(double, double)") {
    test_2<double, double, xtd::atan2, ref_atan2>(values, ulps_double);
  }

  SECTION("double xtd::atan2(int, int)") {
    test_2<double, int, xtd::atan2, ref_atan2>(values, ulps_double);
  }

  SECTION("float xtd::atan2f(float, float)") {
    test_2f<float, float, xtd::atan2f, ref_atan2f>(values, ulps_float);
  }

  SECTION("float xtd::atan2f(double, double)") {
    test_2f<float, double, xtd::atan2f, ref_atan2f>(values, ulps_float);
  }

  SECTION("float xtd::atan2f(int, int)") {
    test_2f<float, int, xtd::atan2f, ref_atan2f>(values, ulps_float);
  }
}
