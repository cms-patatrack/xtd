/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <iostream>
#include <vector>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// SYCL headers
#include <sycl/sycl.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/tan.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 5;
constexpr int ulps_double = 5;

TEST_CASE("xtd::tan", "[tan][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::tan(float)") {
            test<float, float, xtd::tan, mpfr::tan>(queue, values, ulps_float);
          }

          SECTION("double xtd::tan(double)") {
            test<double, double, xtd::tan, mpfr::tan>(queue, values, ulps_double);
          }

          SECTION("double xtd::tan(int)") {
            test<double, int, xtd::tan, mpfr::tan>(queue, values, ulps_double);
          }

          SECTION("float xtd::tanf(float)") {
            test_f<float, float, xtd::tanf, mpfr::tan>(queue, values, ulps_float);
          }

          SECTION("float xtd::tanf(double)") {
            test_f<float, double, xtd::tanf, mpfr::tan>(queue, values, ulps_float);
          }

          SECTION("float xtd::tanf(int)") {
            test_f<float, int, xtd::tanf, mpfr::tan>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
