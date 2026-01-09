/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/lrint.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_lrint.h"

TEST_CASE("xtd::lrint", "[lrint][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("long xtd::lrint(float)") {
          validate<long, float, xtd::lrint, mpfr_lrintf>(device);
        }

        SECTION("long xtd::lrint(double)") {
          validate<long, double, xtd::lrint, mpfr_lrint>(device);
        }

        SECTION("long xtd::lrint(int)") {
          validate<long, int, xtd::lrint, mpfr_lrint>(device);
        }

        SECTION("long xtd::lrintf(float)") {
          validate<long, float, xtd::lrintf, mpfr_lrintf>(device);
        }

        SECTION("long xtd::lrintf(double)") {
          validate<long, double, xtd::lrintf, mpfr_lrintf>(device);
        }

        SECTION("long xtd::lrintf(int)") {
          validate<long, int, xtd::lrintf, mpfr_lrintf>(device);
        }
      }
    }
  }
}
