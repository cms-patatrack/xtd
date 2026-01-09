/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <cassert>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/cpu/name.h"

namespace test::cpu {

  class Device {
  public:
    Device() : name_(cpu_get_name()) {
    }

    ~Device() = default;

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&&) = default;
    Device& operator=(Device&&) = default;

    std::string_view name() const {
      return name_;
    }

  private:
    // Device properties.
    std::string name_;
  };

  inline const Device& device() {
    static Device device;
    return device;
  }

}  // namespace test::cpu
