/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string>

// CUDA headers
#include <cuda_runtime.h>

// test headers
#include "common/cuda/device.h"
#include "common/cuda/version.h"

namespace test::cuda {

  class Platform {
  public:
    Platform() {
      // CUDA version
      name_ = version();

      // CUDA devices
      int count;
      CUDA_CHECK(cudaGetDeviceCount(&count));
      devices_.reserve(count);
      for (int device = 0; device < count; ++device) {
        devices_.emplace_back(device);
      }
    }

    Platform(const Platform&) = delete;
    Platform& operator=(const Platform&) = delete;

    Platform(Platform&&) = default;
    Platform& operator=(Platform&&) = default;

    const std::string& name() const {
      return name_;
    }

    const std::vector<Device>& devices() const {
      return devices_;
    }

  private:
    std::string name_;
    std::vector<Device> devices_;
  };

  inline const Platform& platform() {
    static Platform platform;
    return platform;
  }

}  // namespace test::cuda
