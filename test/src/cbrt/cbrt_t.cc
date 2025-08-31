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
#include "xtd/math/cbrt.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 4;

TEST_CASE("xtd::cbrt", "[cbrt][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::cbrt(float)") {
    test<float, float, xtd::cbrt, mpfr::cbrt>(values, ulps_float);
  }

  SECTION("double xtd::cbrt(double)") {
    test<double, double, xtd::cbrt, mpfr::cbrt>(values, ulps_double);
  }

  SECTION("double xtd::cbrt(int)") {
    test<double, int, xtd::cbrt, mpfr::cbrt>(values, ulps_double);
  }

  SECTION("float xtd::cbrtf(float)") {
    test_f<float, float, xtd::cbrtf, mpfr::cbrt>(values, ulps_float);
  }

  SECTION("float xtd::cbrtf(double)") {
    test_f<float, double, xtd::cbrtf, mpfr::cbrt>(values, ulps_float);
  }

  SECTION("float xtd::cbrtf(int)") {
    test_f<float, int, xtd::cbrtf, mpfr::cbrt>(values, ulps_float);
  }
}
