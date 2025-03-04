
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T asin(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C++ code
    return std::asin(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float asinf(float arg) {
    return asin(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double asinl(long double arg) {
    return asin(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double asinf(T arg) {
    return asin(static_cast<double>(arg));
  }

}  // namespace xtd
