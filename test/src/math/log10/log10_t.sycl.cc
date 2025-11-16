/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/log10.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_log10.h"

constexpr int ulps_single = 3;
constexpr int ulps_double = 3;

TEST_CASE("xtd::log10", "[log10][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::log10(float)") {
            validate<float, float, xtd::log10, mpfr_log10f>(platform, device, ulps_single);
          }

          SECTION("double xtd::log10(double)") {
            validate<double, double, xtd::log10, mpfr_log10>(platform, device, ulps_double);
          }

          SECTION("double xtd::log10(int)") {
            validate<double, int, xtd::log10, mpfr_log10>(platform, device, ulps_double);
          }

          SECTION("float xtd::log10f(float)") {
            validate<float, float, xtd::log10f, mpfr_log10f>(platform, device, ulps_single);
          }

          SECTION("float xtd::log10f(double)") {
            validate<float, double, xtd::log10f, mpfr_log10f>(platform, device, ulps_single);
          }

          SECTION("float xtd::log10f(int)") {
            validate<float, int, xtd::log10f, mpfr_log10f>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
