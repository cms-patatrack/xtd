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
#include "xtd/math/tan.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::tan", "[tan][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::tan(float)") {
    test<float, float, xtd::tan, mpfr::tan>(values, ulps_float);
  }

  SECTION("double xtd::tan(double)") {
    test<double, double, xtd::tan, mpfr::tan>(values, ulps_double);
  }

  SECTION("double xtd::tan(int)") {
    test<double, int, xtd::tan, mpfr::tan>(values, ulps_double);
  }

  SECTION("float xtd::tanf(float)") {
    test_f<float, float, xtd::tanf, mpfr::tan>(values, ulps_float);
  }

  SECTION("float xtd::tanf(double)") {
    test_f<float, double, xtd::tanf, mpfr::tan>(values, ulps_float);
  }

  SECTION("float xtd::tanf(int)") {
    test_f<float, int, xtd::tanf, mpfr::tan>(values, ulps_float);
  }
}
