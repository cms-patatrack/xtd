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
#include "xtd/math/remainder.h"

// test headers
#include "common/sycl_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_remainder = [](mpfr_double y, mpfr_double x) { return mpfr::remainder(y, x); };
constexpr auto ref_remainderf = [](mpfr_single y, mpfr_single x) { return mpfr::remainder(y, x); };

TEST_CASE("xtd::remainder", "[remainder][sycl]") {
  std::vector<double> values = generate_input_values();

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          try {
            sycl::queue queue{device, sycl::property::queue::in_order()};

            SECTION("float xtd::remainder(float, float)") {
              test_2<float, float, xtd::remainder, ref_remainder>(queue, values, ulps_float);
            }

            SECTION("double xtd::remainder(double, double)") {
              test_2<double, double, xtd::remainder, ref_remainder>(queue, values, ulps_double);
            }

            SECTION("double xtd::remainder(int, int)") {
              test_2<double, int, xtd::remainder, ref_remainder>(queue, values, ulps_double);
            }

            SECTION("float xtd::remainderf(float, float)") {
              test_2f<float, float, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
            }

            SECTION("float xtd::remainderf(double, double)") {
              test_2f<float, double, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
            }

            SECTION("float xtd::remainderf(int, int)") {
              test_2f<float, int, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
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
