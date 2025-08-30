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
#include "xtd/math/trunc.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::trunc", "[trunc][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::trunc(float)") {
    test<float, float, xtd::trunc, mpfr::trunc>(values, ulps_float);
  }

  SECTION("double xtd::trunc(double)") {
    test<double, double, xtd::trunc, mpfr::trunc>(values, ulps_double);
  }

  SECTION("double xtd::trunc(int)") {
    test<double, int, xtd::trunc, mpfr::trunc>(values, ulps_double);
  }

  SECTION("float xtd::truncf(float)") {
    test_f<float, float, xtd::truncf, mpfr::trunc>(values, ulps_float);
  }

  SECTION("float xtd::truncf(double)") {
    test_f<float, double, xtd::truncf, mpfr::trunc>(values, ulps_float);
  }

  SECTION("float xtd::truncf(int)") {
    test_f<float, int, xtd::truncf, mpfr::trunc>(values, ulps_float);
  }
}
