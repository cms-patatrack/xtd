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
#include "xtd/math/cosh.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::cosh", "[cosh][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          if (not device.has(sycl::aspect::fp64)) {
            std::cout << "The device " << device.get_info<sycl::info::device::name>()
                      << " does not support double precision floating point operations, some tests will be skipped.\n";
          }
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::cosh(float)") {
            test<float, float, xtd::cosh, mpfr::cosh>(queue, values, ulps_float);
          }

          SECTION("double xtd::cosh(double)") {
            test<double, double, xtd::cosh, mpfr::cosh>(queue, values, ulps_double);
          }

          SECTION("double xtd::cosh(int)") {
            test<double, int, xtd::cosh, mpfr::cosh>(queue, values, ulps_double);
          }

          SECTION("float xtd::coshf(float)") {
            test_f<float, float, xtd::coshf, mpfr::cosh>(queue, values, ulps_float);
          }

          SECTION("float xtd::coshf(double)") {
            test_f<float, double, xtd::coshf, mpfr::cosh>(queue, values, ulps_float);
          }

          SECTION("float xtd::coshf(int)") {
            test_f<float, int, xtd::coshf, mpfr::cosh>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
