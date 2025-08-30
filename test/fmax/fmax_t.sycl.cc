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
#include "xtd/math/fmax.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmax = [](mpfr_double y, mpfr_double x) { return mpfr::fmax(y, x); };
constexpr auto ref_fmaxf = [](mpfr_single y, mpfr_single x) { return mpfr::fmax(y, x); };

TEST_CASE("xtd::fmax", "[fmax][sycl]") {
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

          SECTION("float xtd::fmax(float, float)") {
            test_2<float, float, xtd::fmax, ref_fmax>(queue, values, ulps_float);
          }

          SECTION("double xtd::fmax(double, double)") {
            test_2<double, double, xtd::fmax, ref_fmax>(queue, values, ulps_double);
          }

          SECTION("double xtd::fmax(int, int)") {
            test_2<double, int, xtd::fmax, ref_fmax>(queue, values, ulps_double);
          }

          SECTION("float xtd::fmaxf(float, float)") {
            test_2f<float, float, xtd::fmaxf, ref_fmaxf>(queue, values, ulps_float);
          }

          SECTION("float xtd::fmaxf(double, double)") {
            test_2f<float, double, xtd::fmaxf, ref_fmaxf>(queue, values, ulps_float);
          }

          SECTION("float xtd::fmaxf(int, int)") {
            test_2f<float, int, xtd::fmaxf, ref_fmaxf>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
