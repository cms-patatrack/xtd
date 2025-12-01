/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/llround.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_llround.h"

TEST_CASE("xtd::llround", "[llround][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("long long xtd::llround(float)") {
          validate<long long, float, xtd::llround, mpfr_llroundf>(device);
        }

        SECTION("long long xtd::llround(double)") {
          validate<long long, double, xtd::llround, mpfr_llround>(device);
        }

        SECTION("long long xtd::llround(int)") {
          validate<long long, int, xtd::llround, mpfr_llround>(device);
        }

        SECTION("long long xtd::llroundf(float)") {
          validate<long long, float, xtd::llroundf, mpfr_llroundf>(device);
        }

        SECTION("long long xtd::llroundf(double)") {
          validate<long long, double, xtd::llroundf, mpfr_llroundf>(device);
        }

        SECTION("long long xtd::llroundf(int)") {
          validate<long long, int, xtd::llroundf, mpfr_llroundf>(device);
        }
      }
    }
  }
}
