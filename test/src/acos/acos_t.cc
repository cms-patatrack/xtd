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
#include "xtd/math/acos.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::acos", "[acos][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::acos(float)") {
    test<float, float, xtd::acos, mpfr::acos>(values, ulps_float);
  }

  SECTION("double xtd::acos(double)") {
    test<double, double, xtd::acos, mpfr::acos>(values, ulps_double);
  }

  SECTION("double xtd::acos(int)") {
    test<double, int, xtd::acos, mpfr::acos>(values, ulps_double);
  }

  SECTION("float xtd::acosf(float)") {
    test_f<float, float, xtd::acosf, mpfr::acos>(values, ulps_float);
  }

  SECTION("float xtd::acosf(double)") {
    test_f<float, double, xtd::acosf, mpfr::acos>(values, ulps_float);
  }

  SECTION("float xtd::acosf(int)") {
    test_f<float, int, xtd::acosf, mpfr::acos>(values, ulps_float);
  }
}
