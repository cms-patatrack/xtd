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

// CUDA headers
#include <cuda_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/remainder.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/cuda_version.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_remainder = [](mpfr_double y, mpfr_double x) -> mpfr_double { return mpfr::remainder(y, x); };
constexpr auto ref_remainderf = [](mpfr_single y, mpfr_single x) -> mpfr_single { return mpfr::remainder(y, x); };

TEST_CASE("xtd::remainder", "[remainder][cuda]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("CUDA platform: " << cuda_version()) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      cudaDeviceProp properties;
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("CUDA device " << device << ": " << properties.name) {
        // set the current GPU
        CUDA_CHECK(cudaSetDevice(device));

        // create a CUDA stream for all the asynchronous operations on this GPU
        cudaStream_t queue;
        CUDA_CHECK(cudaStreamCreate(&queue));

        SECTION("float xtd::remainder(float, float)") {
          test_2<float, float, xtd::remainder, ref_remainder>(queue, values, ulps_float);
        }

        SECTION("double xtd::remainder(double, double)") {
          test_2<double, double, xtd::remainder, ref_remainder>(queue, values, ulps_double);
        }

        SECTION("double xtd::remainder(int, int)") {
          test_2<double, int, xtd::remainder, ref_remainder>(queue, values, ulps_double);
        }

        SECTION("float xtd::remainderf(float, float)") {
          test_2f<float, float, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
        }

        SECTION("float xtd::remainderf(double, double)") {
          test_2f<float, double, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
        }

        SECTION("float xtd::remainderf(int, int)") {
          test_2f<float, int, xtd::remainderf, ref_remainderf>(queue, values, ulps_float);
        }

        CUDA_CHECK(cudaStreamDestroy(queue));
      }
    }
  }
}
