/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/log1p.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_log1p.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log1p", "[log1p][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::log1p(float)") {
            validate<float, float, xtd::log1p, mpfr_log1pf>(platform, device, ulps_single);
          }

          SECTION("double xtd::log1p(double)") {
            validate<double, double, xtd::log1p, mpfr_log1p>(platform, device, ulps_double);
          }

          SECTION("double xtd::log1p(int)") {
            validate<double, int, xtd::log1p, mpfr_log1p>(platform, device, ulps_double);
          }

          SECTION("float xtd::log1pf(float)") {
            validate<float, float, xtd::log1pf, mpfr_log1pf>(platform, device, ulps_single);
          }

          SECTION("float xtd::log1pf(double)") {
            validate<float, double, xtd::log1pf, mpfr_log1pf>(platform, device, ulps_single);
          }

          SECTION("float xtd::log1pf(int)") {
            validate<float, int, xtd::log1pf, mpfr_log1pf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
