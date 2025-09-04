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
#include "xtd/math/asin.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::asin", "[asin][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          std::string id;
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::asin(float)") {
            test<float, float, xtd::asin, mpfr::asin>(queue, values, ulps_float);
          }

          SECTION("double xtd::asin(double)") {
            test<double, double, xtd::asin, mpfr::asin>(queue, values, ulps_double);
          }

          SECTION("double xtd::asin(int)") {
            test<double, int, xtd::asin, mpfr::asin>(queue, values, ulps_double);
          }

          SECTION("float xtd::asinf(float)") {
            test_f<float, float, xtd::asinf, mpfr::asin>(queue, values, ulps_float);
          }

          SECTION("float xtd::asinf(double)") {
            test_f<float, double, xtd::asinf, mpfr::asin>(queue, values, ulps_float);
          }

          SECTION("float xtd::asinf(int)") {
            test_f<float, int, xtd::asinf, mpfr::asin>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
