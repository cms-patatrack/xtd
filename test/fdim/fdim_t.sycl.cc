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
#include "xtd/math/fdim.h"

// test headers
#include "common/sycl_test2.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fdim = [](mpfr_double y, mpfr_double x) { return mpfr::fdim(y, x); };
constexpr auto ref_fdimf = [](mpfr_single y, mpfr_single x) { return mpfr::fdim(y, x); };

TEST_CASE("xtd::fdim", "[fdim][sycl]") {
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

          SECTION("float xtd::fdim(float, float)") {
            test_2<float, float, xtd::fdim, ref_fdim>(queue, values, ulps_float);
          }

          SECTION("double xtd::fdim(double, double)") {
            test_2<double, double, xtd::fdim, ref_fdim>(queue, values, ulps_double);
          }

          SECTION("double xtd::fdim(int, int)") {
            test_2<double, int, xtd::fdim, ref_fdim>(queue, values, ulps_double);
          }

          SECTION("float xtd::fdimf(float, float)") {
            test_2f<float, float, xtd::fdimf, ref_fdimf>(queue, values, ulps_float);
          }

          SECTION("float xtd::fdimf(double, double)") {
            test_2f<float, double, xtd::fdimf, ref_fdimf>(queue, values, ulps_float);
          }

          SECTION("float xtd::fdimf(int, int)") {
            test_2f<float, int, xtd::fdimf, ref_fdimf>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
