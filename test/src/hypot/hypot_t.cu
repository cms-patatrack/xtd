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
#include "xtd/math/hypot.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/cuda_version.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 2;

constexpr auto ref_hypot = [](mpfr_double y, mpfr_double x) -> mpfr_double { return mpfr::hypot(y, x); };
constexpr auto ref_hypotf = [](mpfr_single y, mpfr_single x) -> mpfr_single { return mpfr::hypot(y, x); };

TEST_CASE("xtd::hypot", "[hypot][cuda]") {
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

        SECTION("float xtd::hypot(float, float)") {
          test_2<float, float, xtd::hypot, ref_hypot>(queue, values, ulps_float);
        }

        SECTION("double xtd::hypot(double, double)") {
          test_2<double, double, xtd::hypot, ref_hypot>(queue, values, ulps_double);
        }

        SECTION("double xtd::hypot(int, int)") {
          test_2<double, int, xtd::hypot, ref_hypot>(queue, values, ulps_double);
        }

        SECTION("float xtd::hypotf(float, float)") {
          test_2f<float, float, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
        }

        SECTION("float xtd::hypotf(double, double)") {
          test_2f<float, double, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
        }

        SECTION("float xtd::hypotf(int, int)") {
          test_2f<float, int, xtd::hypotf, ref_hypotf>(queue, values, ulps_float);
        }

        CUDA_CHECK(cudaStreamDestroy(queue));
      }
    }
  }
}
