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
#include "xtd/math/cosh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cosh", "[cosh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::cosh(float)") {
    test<float, float, xtd::cosh, mpfr::cosh>(values, ulps_float);
  }

  SECTION("double xtd::cosh(double)") {
    test<double, double, xtd::cosh, mpfr::cosh>(values, ulps_double);
  }

  SECTION("double xtd::cosh(int)") {
    test<double, int, xtd::cosh, mpfr::cosh>(values, ulps_double);
  }

  SECTION("float xtd::coshf(float)") {
    test_f<float, float, xtd::coshf, mpfr::cosh>(values, ulps_float);
  }

  SECTION("float xtd::coshf(double)") {
    test_f<float, double, xtd::coshf, mpfr::cosh>(values, ulps_float);
  }

  SECTION("float xtd::coshf(int)") {
    test_f<float, int, xtd::coshf, mpfr::cosh>(values, ulps_float);
  }
}
