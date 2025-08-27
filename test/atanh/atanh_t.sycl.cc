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
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// SYCL headers
#include <sycl/sycl.hpp>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/atanh.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 5;
constexpr int ulps_double = 5;

TEST_CASE("xtd::atanh", "[atanh][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          try {
            sycl::queue queue{device, sycl::property::queue::in_order()};

            SECTION("float xtd::atanh(float)") {
              test<float, float, xtd::atanh, mpfr::atanh>(queue, values, ulps_float);
            }

            SECTION("double xtd::atanh(double)") {
              test<double, double, xtd::atanh, mpfr::atanh>(queue, values, ulps_double);
            }

            SECTION("double xtd::atanh(int)") {
              test<double, int, xtd::atanh, mpfr::atanh>(queue, values, ulps_double);
            }

            SECTION("float xtd::atanhf(float)") {
              test_f<float, float, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
            }

            SECTION("float xtd::atanhf(double)") {
              test_f<float, double, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
            }

            SECTION("float xtd::atanhf(int)") {
              test_f<float, int, xtd::atanhf, mpfr::atanh>(queue, values, ulps_float);
            }

          } catch (sycl::exception const &e) {
            std::cerr << "SYCL exception:\n"
                      << e.what() << "\ncaught while running on platform "
                      << platform.get_info<sycl::info::platform::name>() << ", device "
                      << device.get_info<sycl::info::device::name>() << '\n';
            continue;
          }
        }
      }
    }
  }
}
