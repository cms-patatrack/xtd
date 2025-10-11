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
#include "xtd/math/fmod.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::fmod(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::fmod(x, y); };

TEST_CASE("xtd::fmod", "[fmod][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          std::string id;
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::fmod(float, float)") {
            test_aa<float, float, xtd::fmod, ref_function>(queue, values, ulps_single);
          }

          SECTION("double xtd::fmod(double, double)") {
            test_aa<double, double, xtd::fmod, ref_function>(queue, values, ulps_double);
          }

          SECTION("double xtd::fmod(int, int)") {
            test_aa<double, int, xtd::fmod, ref_function>(queue, values, ulps_double);
          }

          SECTION("float xtd::fmodf(float, float)") {
            test_ff<float, float, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::fmodf(double, double)") {
            test_ff<float, double, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
          }

          SECTION("float xtd::fmodf(int, int)") {
            test_ff<float, int, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
          }
        }
      }
    }
  }
}
