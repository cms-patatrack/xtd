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
#include "xtd/math/acos.h"

// test headers
#include "common/sycl_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::acos", "[acos][sycl]") {
  std::vector<double> values = generate_input_values();

  int pid = 0;
  for (const auto &platform : sycl::platform::get_platforms()) {
    DYNAMIC_SECTION("SYCL platform " << ++pid << ": " << platform.get_info<sycl::info::platform::name>()) {
      int id = 0;
      for (const auto &device : platform.get_devices()) {
        DYNAMIC_SECTION("SYCL device " << pid << '.' << ++id << ": " << device.get_info<sycl::info::device::name>()) {
          std::string id;
          sycl::queue queue{device, sycl::property::queue::in_order()};

          SECTION("float xtd::acos(float)") {
            test<float, float, xtd::acos, mpfr::acos>(queue, values, ulps_float);
          }

          SECTION("double xtd::acos(double)") {
            test<double, double, xtd::acos, mpfr::acos>(queue, values, ulps_double);
          }

          SECTION("double xtd::acos(int)") {
            test<double, int, xtd::acos, mpfr::acos>(queue, values, ulps_double);
          }

          SECTION("float xtd::acosf(float)") {
            test_f<float, float, xtd::acosf, mpfr::acos>(queue, values, ulps_float);
          }

          SECTION("float xtd::acosf(double)") {
            test_f<float, double, xtd::acosf, mpfr::acos>(queue, values, ulps_float);
          }

          SECTION("float xtd::acosf(int)") {
            test_f<float, int, xtd::acosf, mpfr::acos>(queue, values, ulps_float);
          }
        }
      }
    }
  }
}
