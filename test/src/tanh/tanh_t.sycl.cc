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
#include "xtd/math/tanh.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 5;
constexpr int ulps_double = 5;

TEST_CASE("xtd::tanh", "[tanh][sycl]") {
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

          SECTION("float xtd::tanh(float)") {
            test<float, float, xtd::tanh, mpfr::tanh>(queue, values, ulps_float);
          }

          SECTION("double xtd::tanh(double)") {
            test<double, double, xtd::tanh, mpfr::tanh>(queue, values, ulps_double);
          }

          SECTION("double xtd::tanh(int)") {
            test<double, int, xtd::tanh, mpfr::tanh>(queue, values, ulps_double);
          }

          SECTION("float xtd::tanhf(float)") {
            test_f<float, float, xtd::tanhf, mpfr::tanh>(queue, values, ulps_float);
          }

          SECTION("float xtd::tanhf(double)") {
            test_f<float, double, xtd::tanhf, mpfr::tanh>(queue, values, ulps_float);
          }

          SECTION("float xtd::tanhf(int)") {
            test_f<float, int, xtd::tanhf, mpfr::tanh>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
