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
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// CUDA headers
#include <cuda_runtime.h>

// xtd headers
#include "math/sin.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"

TEST_CASE("xtd::sin", "[sin][cuda]") {
  int deviceCount;
  cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped." << std::endl;
    exit(EXIT_SUCCESS);
  }

  std::vector<double> values{-1., 0., M_PI / 2, M_PI, 42.};

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

      SECTION("float xtd::sin(float)") {
        test<float, float, xtd::sin, std::sin>(queue, values);
      }

      SECTION("double xtd::sin(double)") {
        test<double, double, xtd::sin, std::sin>(queue, values);
      }

      SECTION("double xtd::sin(int)") {
        test<double, int, xtd::sin, std::sin>(queue, values);
      }

      // Note: GCC prior to v14.1 and clang prior to v19.1 do not provide std::sinf().
      // As a workarund, use C sinf().

      SECTION("float xtd::sinf(float)") {
        test_f<float, float, xtd::sinf, ::sinf>(queue, values);
      }

      SECTION("float xtd::sinf(double)") {
        test_f<float, double, xtd::sinf, ::sinf>(queue, values);
      }

      SECTION("float xtd::sinf(int)") {
        test_f<float, int, xtd::sinf, ::sinf>(queue, values);
      }

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
