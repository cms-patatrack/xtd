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
#include "xtd/math/acosh.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::acosh(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::acosh(x); };

TEST_CASE("xtd::acosh", "[acosh][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          std::string id;
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::acosh(float)") {
            test_a<float, float, xtd::acosh, ref_function>(queue, values, ulps_single);
          }

          SECTION("double xtd::acosh(double)") {
            test_a<double, double, xtd::acosh, ref_function>(queue, values, ulps_double);
          }

          SECTION("double xtd::acosh(int)") {
            test_a<double, int, xtd::acosh, ref_function>(queue, values, ulps_double);
          }

          SECTION("float xtd::acoshf(float)") {
            test_f<float, float, xtd::acoshf, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::acoshf(double)") {
            test_f<float, double, xtd::acoshf, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::acoshf(int)") {
            test_f<float, int, xtd::acoshf, ref_functionf>(queue, values, ulps_single);
          }
        }
      }
    }
  }
}
