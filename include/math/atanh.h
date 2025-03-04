
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T atanh(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atanh(arg);
#else
    // standard C++ code
    return std::atanh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float atanhf(float arg) {
    return atanh(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double atanhl(long double arg) {
    return atanh(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double atanh(T arg) {
    return atanh(static_cast<double>(arg));
  }

}  // namespace xtd
