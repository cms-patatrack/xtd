/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <iostream>
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
#include "xtd/math/fmod.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmod = [](mpfr_double y, mpfr_double x) -> mpfr_double { return mpfr::fmod(y, x); };
constexpr auto ref_fmodf = [](mpfr_single y, mpfr_single x) -> mpfr_single { return mpfr::fmod(y, x); };

TEST_CASE("xtd::fmod", "[fmod][cuda]") {
  std::vector<double> values = generate_input_values();

  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

  for (int device = 0; device < deviceCount; ++device) {
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::string section = "CUDA GPU "s + std::to_string(device) + ": "s + properties.name;
    SECTION(section) {
      // set the current GPU
      CUDA_CHECK(cudaSetDevice(device));

      // create a CUDA stream for all the asynchronous operations on this GPU
      cudaStream_t queue;
      CUDA_CHECK(cudaStreamCreate(&queue));

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

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
