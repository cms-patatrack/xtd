/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/sqrt.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::sqrt(float)") {
    test<float, float, xtd::sqrt, mpfr::sqrt>(values, ulps_float);
  }

  SECTION("double xtd::sqrt(double)") {
    test<double, double, xtd::sqrt, mpfr::sqrt>(values, ulps_double);
  }

  SECTION("double xtd::sqrt(int)") {
    test<double, int, xtd::sqrt, mpfr::sqrt>(values, ulps_double);
  }

  SECTION("float xtd::sqrtf(float)") {
    test_f<float, float, xtd::sqrtf, mpfr::sqrt>(values, ulps_float);
  }

  SECTION("float xtd::sqrtf(double)") {
    test_f<float, double, xtd::sqrtf, mpfr::sqrt>(values, ulps_float);
  }

  SECTION("float xtd::sqrtf(int)") {
    test_f<float, int, xtd::sqrtf, mpfr::sqrt>(values, ulps_float);
  }
}
