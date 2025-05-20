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

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/cpu_test.h"

TEST_CASE("xtd::sin", "[sin][cpu]") {
  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  SECTION("float xtd::sin(float)") {
    test<float, float, xtd::sin, std::sin>(values);
  }

  SECTION("double xtd::sin(double)") {
    test<double, double, xtd::sin, std::sin>(values);
  }

  SECTION("double xtd::sin(int)") {
    test<double, int, xtd::sin, std::sin>(values);
  }

  // Note: GCC prior to v14.1 and clang prior to v19.1 do not provide std::sinf().
  // As a workarund, use C sinf().

  SECTION("float xtd::sinf(float)") {
    test_f<float, float, xtd::sinf, ::sinf>(values);
  }

  SECTION("float xtd::sinf(double)") {
    test_f<float, double, xtd::sinf, ::sinf>(values);
  }

  SECTION("float xtd::sinf(int)") {
    test_f<float, int, xtd::sinf, ::sinf>(values);
  }
}
