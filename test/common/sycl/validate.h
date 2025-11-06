/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

// SYCL headers
#include <sycl/sycl.hpp>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// test headers
#include "common/compare.h"
#include "common/halton.h"
#include "common/mpfr.h"
#include "common/sycl/device.h"
#include "common/sycl/inputs.h"
#include "common/sycl/platform.h"

namespace test::sycl {

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << val.value_ << " ["
           << std::hexfloat << val.value_ << "]";
    out << buffer.str();
    return out;
  }

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType),
            ResultType (*RefFunc)(InputType)>
  inline void validate(const Platform& platform, const Device& device, int ulps = 0) try {
    if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) { result_d[t] = XtdFunc(values_d[t]); });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      InputType x = values_h[t];
      // Read the result of the xtd function.
      ResultType result = result_h[t];
      // Compare the result with the reference.
      ResultType reference = RefFunc(x);
      // Log the comparison.
      INFO("index: " << t << "\ninput:      " << detailed(x) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      compare(result, reference, ulps);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType),
            ResultType (*RefFunc)(InputType)>
  inline void validate(const Platform& platform, const Device& device) try {
    if constexpr (std::is_same_v<InputType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) { result_d[t] = XtdFunc(values_d[t]); });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      InputType x = values_h[t];
      // Read the result of the xtd function.
      ResultType result = result_h[t];
      // Compare the result with the reference.
      ResultType reference = RefFunc(x);
      // Log the comparison.
      INFO("index: " << t << "\ninput:      " << detailed(x) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      compare(result, reference);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType)>
  inline void validate(const Platform& platform, const Device& device, int ulps = 0) try {
    if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) {
        // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size).
        auto [i, j] = halton<2>(static_cast<size_t>(t), size);
        result_d[t] = static_cast<ResultType>(XtdFunc(values_d[i], values_d[j]));
      });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size).
      auto [i, j] = halton<2>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference, ulps);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType)>
  inline void validate(const Platform& platform, const Device& device) try {
    if constexpr (std::is_same_v<InputType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) {
        // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size).
        auto [i, j] = halton<2>(static_cast<size_t>(t), size);
        result_d[t] = static_cast<ResultType>(XtdFunc(values_d[i], values_d[j]));
      });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size).
      auto [i, j] = halton<2>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate(const Platform& platform, const Device& device, int ulps = 0) try {
    if constexpr (std::is_same_v<InputType, double> or std::is_same_v<ResultType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) {
        // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size).
        auto [i, j, k] = halton<3>(static_cast<size_t>(t), size);
        result_d[t] = static_cast<ResultType>(XtdFunc(values_d[i], values_d[j], values_d[k]));
      });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size).
      auto [i, j, k] = halton<3>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      InputType z = values_h[k];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y, z);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference, ulps);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate(const Platform& platform, const Device& device) try {
    if constexpr (std::is_same_v<InputType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, 0);
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) {
        // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size).
        auto [i, j, k] = halton<3>(static_cast<size_t>(t), size);
        result_d[t] = static_cast<ResultType>(XtdFunc(values_d[i], values_d[j], values_d[k]));
      });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size).
      auto [i, j, k] = halton<3>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      InputType z = values_h[k];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y, z);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

}  // namespace test::sycl
