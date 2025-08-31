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
#include "xtd/math/sinh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::sinh", "[sinh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::sinh(float)") {
    test<float, float, xtd::sinh, mpfr::sinh>(values, ulps_float);
  }

  SECTION("double xtd::sinh(double)") {
    test<double, double, xtd::sinh, mpfr::sinh>(values, ulps_double);
  }

  SECTION("double xtd::sinh(int)") {
    test<double, int, xtd::sinh, mpfr::sinh>(values, ulps_double);
  }

  SECTION("float xtd::sinhf(float)") {
    test_f<float, float, xtd::sinhf, mpfr::sinh>(values, ulps_float);
  }

  SECTION("float xtd::sinhf(double)") {
    test_f<float, double, xtd::sinhf, mpfr::sinh>(values, ulps_float);
  }

  SECTION("float xtd::sinhf(int)") {
    test_f<float, int, xtd::sinhf, mpfr::sinh>(values, ulps_float);
  }
}
