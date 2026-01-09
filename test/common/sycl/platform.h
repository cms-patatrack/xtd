/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string>
#include <string_view>
#include <vector>

// SYCL headers
#include <sycl/sycl.hpp>

// test headers
#include "common/sycl/device.h"

namespace test::sycl {

  class Platform {
  public:
    Platform(int index, ::sycl::platform platform)
        : index_{index}, platform_{platform}, name_{platform_.get_info<::sycl::info::platform::name>()} {
      std::vector<::sycl::device> devices = platform_.get_devices();
      devices_.reserve(devices.size());
      for (size_t i = 0; i < devices.size(); ++i) {
        devices_.emplace_back(i, devices[i]);
      }
    }

    // Not copyable.
    Platform(const Platform&) = delete;
    Platform& operator=(const Platform&) = delete;

    // Movable.
    Platform(Platform&&) = default;
    Platform& operator=(Platform&&) = default;

    int index() const {
      return index_;
    }

    ::sycl::platform platform() const {
      return platform_;
    }

    const std::string& name() const {
      return name_;
    }

    const std::vector<Device>& devices() const {
      return devices_;
    }

  private:
    int index_;
    ::sycl::platform platform_;
    std::string name_;
    std::vector<Device> devices_;
  };

  inline std::vector<Platform> platforms_impl() {
    const std::vector<::sycl::platform>& sycl_platforms = ::sycl::platform::get_platforms();
    std::vector<Platform> platforms;
    platforms.reserve(sycl_platforms.size());
    for (size_t i = 0; i < sycl_platforms.size(); ++i) {
      platforms.emplace_back(i, sycl_platforms[i]);
    }
    return platforms;
  }

  inline std::vector<Platform>& platforms() {
    static std::vector<Platform> platforms = platforms_impl();
    return platforms;
  }

}  // namespace test::sycl
