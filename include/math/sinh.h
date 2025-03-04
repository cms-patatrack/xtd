#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T sinh(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C++ code
    return std::sinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double sinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C++ code
    return std::sinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float sinhf(float arg) {
    return sinh(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double sinhl(long double arg) {
    return sinh(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double sinh(T arg) {
    return sinh(static_cast<double>(arg));
  }

}  // namespace xtd
