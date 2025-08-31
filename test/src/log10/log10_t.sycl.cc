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
#include "xtd/math/log10.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 3;
constexpr int ulps_double = 3;

TEST_CASE("xtd::log10", "[log10][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::log10(float)") {
            test<float, float, xtd::log10, mpfr::log10>(queue, values, ulps_float);
          }

          SECTION("double xtd::log10(double)") {
            test<double, double, xtd::log10, mpfr::log10>(queue, values, ulps_double);
          }

          SECTION("double xtd::log10(int)") {
            test<double, int, xtd::log10, mpfr::log10>(queue, values, ulps_double);
          }

          SECTION("float xtd::log10f(float)") {
            test_f<float, float, xtd::log10f, mpfr::log10>(queue, values, ulps_float);
          }

          SECTION("float xtd::log10f(double)") {
            test_f<float, double, xtd::log10f, mpfr::log10>(queue, values, ulps_float);
          }

          SECTION("float xtd::log10f(int)") {
            test_f<float, int, xtd::log10f, mpfr::log10>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
