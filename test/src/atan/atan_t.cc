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
#include "xtd/math/atan.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::atan", "[atan][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atan(float)") {
    test<float, float, xtd::atan, mpfr::atan>(values, ulps_float);
  }

  SECTION("double xtd::atan(double)") {
    test<double, double, xtd::atan, mpfr::atan>(values, ulps_double);
  }

  SECTION("double xtd::atan(int)") {
    test<double, int, xtd::atan, mpfr::atan>(values, ulps_double);
  }

  SECTION("float xtd::atanf(float)") {
    test_f<float, float, xtd::atanf, mpfr::atan>(values, ulps_float);
  }

  SECTION("float xtd::atanf(double)") {
    test_f<float, double, xtd::atanf, mpfr::atan>(values, ulps_float);
  }

  SECTION("float xtd::atanf(int)") {
    test_f<float, int, xtd::atanf, mpfr::atan>(values, ulps_float);
  }
}
