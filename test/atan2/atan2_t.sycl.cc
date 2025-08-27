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
#include "xtd/math/atan2.h"

// test headers
#include "common/sycl_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 6;
constexpr int ulps_double = 6;

constexpr auto ref_atan2 = [](mpfr_double y, mpfr_double x) { return mpfr::atan2(y, x); };
constexpr auto ref_atan2f = [](mpfr_single y, mpfr_single x) { return mpfr::atan2(y, x); };

TEST_CASE("xtd::atan2", "[atan2][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          try {
            sycl::queue queue{device, sycl::property::queue::in_order()};

            SECTION("float xtd::atan2(float, float)") {
              test_2<float, float, xtd::atan2, ref_atan2>(queue, values, ulps_float);
            }

            SECTION("double xtd::atan2(double, double)") {
              test_2<double, double, xtd::atan2, ref_atan2>(queue, values, ulps_double);
            }

            SECTION("double xtd::atan2(int, int)") {
              test_2<double, int, xtd::atan2, ref_atan2>(queue, values, ulps_double);
            }

            SECTION("float xtd::atan2f(float, float)") {
              test_2f<float, float, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
            }

            SECTION("float xtd::atan2f(double, double)") {
              test_2f<float, double, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
            }

            SECTION("float xtd::atan2f(int, int)") {
              test_2f<float, int, xtd::atan2f, ref_atan2f>(queue, values, ulps_float);
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
