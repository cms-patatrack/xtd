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
#include "xtd/math/atan2.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 6;
constexpr int ulps_double = 6;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::atan2(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::atan2(x, y); };

TEST_CASE("xtd::atan2", "[atan2][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          std::string id;
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::atan2(float, float)") {
            test_aa<float, float, xtd::atan2, ref_function>(queue, values, ulps_single);
          }

          SECTION("double xtd::atan2(double, double)") {
            test_aa<double, double, xtd::atan2, ref_function>(queue, values, ulps_double);
          }

          SECTION("double xtd::atan2(int, int)") {
            test_aa<double, int, xtd::atan2, ref_function>(queue, values, ulps_double);
          }

          SECTION("float xtd::atan2f(float, float)") {
            test_ff<float, float, xtd::atan2f, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::atan2f(double, double)") {
            test_ff<float, double, xtd::atan2f, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::atan2f(int, int)") {
            test_ff<float, int, xtd::atan2f, ref_functionf>(queue, values, ulps_single);
          }
        }
      }
    }
  }
}
