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
#include "xtd/stdlib/abs.h"

// test headers
#include "common/cpu_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::abs", "[abs][cpu]") {
  std::vector<double> values = generate_input_values();

  SECTION("float xtd::abs(float)") {
    test<float, float, xtd::abs, mpfr::fabs>(values, ulps_float);
  }

  SECTION("double xtd::abs(double)") {
    test<double, double, xtd::abs, mpfr::fabs>(values, ulps_double);
  }

  SECTION("int xtd::abs(int)") {
    test_i<int, xtd::abs, std::abs>(values);
  }

  SECTION("long long xtd::abs(long long)") {
    test_i<long long, xtd::abs, std::abs>(values);
  }
}
