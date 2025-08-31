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
#include "xtd/math/abs.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::abs", "[abs][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::abs(float)") {
            test<float, float, xtd::abs, mpfr::fabs>(queue, values, ulps_float);
          }

          SECTION("double xtd::abs(double)") {
            test<double, double, xtd::abs, mpfr::fabs>(queue, values, ulps_double);
          }

          SECTION("int xtd::abs(int)") {
            test_i<int, xtd::abs, std::abs>(queue, values);
          }

          SECTION("long long xtd::abs(long long)") {
            test_i<long long, xtd::abs, std::abs>(queue, values);
          }
        }
      }
    }
  }
}
