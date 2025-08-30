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
#include "xtd/math/atanh.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atanh", "[atanh][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::atanh(float)") {
    test<float, float, xtd::atanh, mpfr::atanh>(values, ulps_float);
  }

  SECTION("double xtd::atanh(double)") {
    test<double, double, xtd::atanh, mpfr::atanh>(values, ulps_double);
  }

  SECTION("double xtd::atanh(int)") {
    test<double, int, xtd::atanh, mpfr::atanh>(values, ulps_double);
  }

  SECTION("float xtd::atanhf(float)") {
    test_f<float, float, xtd::atanhf, mpfr::atanh>(values, ulps_float);
  }

  SECTION("float xtd::atanhf(double)") {
    test_f<float, double, xtd::atanhf, mpfr::atanh>(values, ulps_float);
  }

  SECTION("float xtd::atanhf(int)") {
    test_f<float, int, xtd::atanhf, mpfr::atanh>(values, ulps_float);
  }
}
