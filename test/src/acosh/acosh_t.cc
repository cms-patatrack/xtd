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

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::acosh", "[acosh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::acosh(float)") {
    test<float, float, xtd::acosh, mpfr::acosh>(values, ulps_float);
  }

  SECTION("double xtd::acosh(double)") {
    test<double, double, xtd::acosh, mpfr::acosh>(values, ulps_double);
  }

  SECTION("double xtd::acosh(int)") {
    test<double, int, xtd::acosh, mpfr::acosh>(values, ulps_double);
  }

  SECTION("float xtd::acoshf(float)") {
    test_f<float, float, xtd::acoshf, mpfr::acosh>(values, ulps_float);
  }

  SECTION("float xtd::acoshf(double)") {
    test_f<float, double, xtd::acoshf, mpfr::acosh>(values, ulps_float);
  }

  SECTION("float xtd::acoshf(int)") {
    test_f<float, int, xtd::acoshf, mpfr::acosh>(values, ulps_float);
  }
}
