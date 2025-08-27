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
#include "xtd/math/hypot.h"

// test headers
#include "common/sycl_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 4;
constexpr int ulps_double = 4;

constexpr auto ref_hypot = [](mpfr_double y, mpfr_double x) { return mpfr::hypot(y, x); };
constexpr auto ref_hypotf = [](mpfr_single y, mpfr_single x) { return mpfr::hypot(y, x); };

TEST_CASE("xtd::hypot", "[hypot][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          try {
            sycl::queue queue{device, sycl::property::queue::in_order()};

            SECTION("float xtd::hypot(float, float)") {
              test_2<float, float, xtd::hypot, ref_hypot>(queue, values, ulps_float);
            }

            SECTION("double xtd::hypot(double, double)") {
              test_2<double, double, xtd::hypot, ref_hypot>(queue, values, ulps_double);
            }

            SECTION("double xtd::hypot(int, int)") {
              test_2<double, int, xtd::hypot, ref_hypot>(queue, values, ulps_double);
            }

            SECTION("float xtd::hypotf(float, float)") {
              test_2f<float, float, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
            }

            SECTION("float xtd::hypotf(double, double)") {
              test_2f<float, double, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
            }

            SECTION("float xtd::hypotf(int, int)") {
              test_2f<float, int, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
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
