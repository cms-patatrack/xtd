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
#include "xtd/math/asinh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::asinh", "[asinh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::asinh(float)") {
    test<float, float, xtd::asinh, mpfr::asinh>(values, ulps_float);
  }

  SECTION("double xtd::asinh(double)") {
    test<double, double, xtd::asinh, mpfr::asinh>(values, ulps_double);
  }

  SECTION("double xtd::asinh(int)") {
    test<double, int, xtd::asinh, mpfr::asinh>(values, ulps_double);
  }

  SECTION("float xtd::asinhf(float)") {
    test_f<float, float, xtd::asinhf, mpfr::asinh>(values, ulps_float);
  }

  SECTION("float xtd::asinhf(double)") {
    test_f<float, double, xtd::asinhf, mpfr::asinh>(values, ulps_float);
  }

  SECTION("float xtd::asinhf(int)") {
    test_f<float, int, xtd::asinhf, mpfr::asinh>(values, ulps_float);
  }
}
