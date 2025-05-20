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

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/sycl_test.h"

TEST_CASE("xtd::sin", "[sin][sycl]") {
  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

  for (const auto &platform : sycl::platform::get_platforms()) {
    SECTION(platform.get_info<sycl::info::platform::name>()) {
      for (const auto &device : platform.get_devices()) {
        SECTION(device.get_info<sycl::info::device::name>()) {
          try {
            sycl::queue queue{device, sycl::property::queue::in_order()};

            SECTION("float xtd::sin(float)") {
              test<float, float, xtd::sin, std::sin>(queue, values);
            }

            SECTION("double xtd::sin(double)") {
              test<double, double, xtd::sin, std::sin>(queue, values);
            }

            SECTION("double xtd::sin(int)") {
              test<double, int, xtd::sin, std::sin>(queue, values);
            }

            // Note: clang prior to v19.1 does not provide std::sinf().
            // As a workarund, use C sinf().

            SECTION("float xtd::sinf(float)") {
              test_f<float, float, xtd::sinf, ::sinf>(queue, values);
            }

            SECTION("float xtd::sinf(double)") {
              test_f<float, double, xtd::sinf, ::sinf>(queue, values);
            }

            SECTION("float xtd::sinf(int)") {
              test_f<float, int, xtd::sinf, ::sinf>(queue, values);
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
