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
#include "xtd/stdlib/abs.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::fabs(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::fabs(x); };

TEST_CASE("xtd::abs", "[abs][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::abs(float)") {
            test_f<float, float, xtd::abs, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("double xtd::abs(double)") {
            test_a<double, double, xtd::abs, ref_function>(queue, values, ulps_double);
          }

          SECTION("int xtd::abs(int)") {
            test_i<int, xtd::abs, std::abs>(queue, values);
          }

          SECTION("long xtd::abs(long)") {
            test_i<long, xtd::abs, std::abs>(queue, values);
          }

          SECTION("long long xtd::abs(long long)") {
            test_i<long long, xtd::abs, std::abs>(queue, values);
          }
        }
      }
    }
  }
}
