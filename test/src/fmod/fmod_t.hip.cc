/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <string>
#include <vector>
using namespace std::literals;

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/fmod.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/hip_version.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmod = [](mpfr_double y, mpfr_double x) { return mpfr::fmod(y, x); };
constexpr auto ref_fmodf = [](mpfr_single y, mpfr_single x) { return mpfr::fmod(y, x); };

TEST_CASE("xtd::fmod", "[fmod][hip]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("HIP platform: " << hip_version()) {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      hipDeviceProp_t properties;
      HIP_CHECK(hipGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("HIP device " << device << ": " << properties.name) {
        // set the current GPU
        HIP_CHECK(hipSetDevice(device));

        // create a HIP stream for all the asynchronous operations on this GPU
        hipStream_t queue;
        HIP_CHECK(hipStreamCreate(&queue));

        SECTION("float xtd::fmod(float, float)") {
          test_2<float, float, xtd::fmod, ref_fmod>(queue, values, ulps_float);
        }

        SECTION("double xtd::fmod(double, double)") {
          test_2<double, double, xtd::fmod, ref_fmod>(queue, values, ulps_double);
        }

        SECTION("double xtd::fmod(int, int)") {
          test_2<double, int, xtd::fmod, ref_fmod>(queue, values, ulps_double);
        }

        SECTION("float xtd::fmodf(float, float)") {
          test_2f<float, float, xtd::fmodf, ref_fmodf>(queue, values, ulps_float);
        }

        SECTION("float xtd::fmodf(double, double)") {
          test_2f<float, double, xtd::fmodf, ref_fmodf>(queue, values, ulps_float);
        }

        SECTION("float xtd::fmodf(int, int)") {
          test_2f<float, int, xtd::fmodf, ref_fmodf>(queue, values, ulps_float);
        }

        HIP_CHECK(hipStreamDestroy(queue));
      }
    }
  }
}
