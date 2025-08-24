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
#include "xtd/math/sin.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::sin", "[sin][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::sin(float)") {
    test<float, float, xtd::sin, mpfr::sin>(values, ulps_float);
  }

  SECTION("double xtd::sin(double)") {
    test<double, double, xtd::sin, mpfr::sin>(values, ulps_double);
  }

  SECTION("double xtd::sin(int)") {
    test<double, int, xtd::sin, mpfr::sin>(values, ulps_double);
  }

  SECTION("float xtd::sinf(float)") {
    test_f<float, float, xtd::sinf, mpfr::sin>(values, ulps_float);
  }

  SECTION("float xtd::sinf(double)") {
    test_f<float, double, xtd::sinf, mpfr::sin>(values, ulps_float);
  }

  SECTION("float xtd::sinf(int)") {
    test_f<float, int, xtd::sinf, mpfr::sin>(values, ulps_float);
  }
}
