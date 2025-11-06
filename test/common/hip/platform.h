/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string>

// HIP headers
#include <hip/hip_runtime.h>

// test headers
#include "common/hip/device.h"
#include "common/hip/version.h"

namespace test::hip {

  class Platform {
  public:
    Platform() {
      // HIP version
      name_ = version();

      // HIP devices
      int count;
      HIP_CHECK(hipGetDeviceCount(&count));
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

}  // namespace test::hip
