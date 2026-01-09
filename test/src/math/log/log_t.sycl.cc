/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/log.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_log.h"

constexpr int ulps_single = 3;
constexpr int ulps_double = 3;

TEST_CASE("xtd::log", "[log][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::log(float)") {
            validate<float, float, xtd::log, mpfr_logf>(platform, device, ulps_single);
          }

          SECTION("double xtd::log(double)") {
            validate<double, double, xtd::log, mpfr_log>(platform, device, ulps_double);
          }

          SECTION("double xtd::log(int)") {
            validate<double, int, xtd::log, mpfr_log>(platform, device, ulps_double);
          }

          SECTION("float xtd::logf(float)") {
            validate<float, float, xtd::logf, mpfr_logf>(platform, device, ulps_single);
          }

          SECTION("float xtd::logf(double)") {
            validate<float, double, xtd::logf, mpfr_logf>(platform, device, ulps_single);
          }

          SECTION("float xtd::logf(int)") {
            validate<float, int, xtd::logf, mpfr_logf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
