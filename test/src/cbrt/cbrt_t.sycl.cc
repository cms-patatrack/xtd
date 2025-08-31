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
#include "xtd/math/cbrt.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cbrt", "[cbrt][sycl]") {
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

          SECTION("float xtd::cbrt(float)") {
            test<float, float, xtd::cbrt, mpfr::cbrt>(queue, values, ulps_float);
          }

          SECTION("double xtd::cbrt(double)") {
            test<double, double, xtd::cbrt, mpfr::cbrt>(queue, values, ulps_double);
          }

          SECTION("double xtd::cbrt(int)") {
            test<double, int, xtd::cbrt, mpfr::cbrt>(queue, values, ulps_double);
          }

          SECTION("float xtd::cbrtf(float)") {
            test_f<float, float, xtd::cbrtf, mpfr::cbrt>(queue, values, ulps_float);
          }

          SECTION("float xtd::cbrtf(double)") {
            test_f<float, double, xtd::cbrtf, mpfr::cbrt>(queue, values, ulps_float);
          }

          SECTION("float xtd::cbrtf(int)") {
            test_f<float, int, xtd::cbrtf, mpfr::cbrt>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
