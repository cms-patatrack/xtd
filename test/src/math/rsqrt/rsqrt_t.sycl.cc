/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/rsqrt.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_rsqrt.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::rsqrt", "[rsqrt][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::rsqrt(float)") {
            validate<float, float, xtd::rsqrt, mpfr_rsqrtf>(platform, device, ulps_single);
          }

          SECTION("double xtd::rsqrt(double)") {
            validate<double, double, xtd::rsqrt, mpfr_rsqrt>(platform, device, ulps_double);
          }

          SECTION("double xtd::rsqrt(int)") {
            validate<double, int, xtd::rsqrt, mpfr_rsqrt>(platform, device, ulps_double);
          }

          SECTION("float xtd::rsqrtf(float)") {
            validate<float, float, xtd::rsqrtf, mpfr_rsqrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::rsqrtf(double)") {
            validate<float, double, xtd::rsqrtf, mpfr_rsqrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::rsqrtf(int)") {
            validate<float, int, xtd::rsqrtf, mpfr_rsqrtf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
