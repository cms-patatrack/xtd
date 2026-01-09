/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string>
#include <string_view>

// SYCL headers
#include <sycl/sycl.hpp>

namespace test::sycl {

  class Device {
  public:
    Device(int index, ::sycl::device device)
        : index_{index},
          device_{device},
          queue_{device_, ::sycl::property::queue::in_order()},
          name_{device_.get_info<::sycl::info::device::name>()} {
    }

    ~Device() {
      queue_.wait();
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&&) = default;
    Device& operator=(Device&&) = default;

    int index() const {
      return index_;
    }

    std::string_view name() const {
      return name_;
    }

    ::sycl::device device() const {
      return device_;
    }

    ::sycl::queue queue() const {
      return queue_;
    }

  private:
    int index_;
    ::sycl::device device_;
    ::sycl::queue queue_;
    std::string name_;
  };

}  // namespace test::sycl
