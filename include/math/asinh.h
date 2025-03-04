
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr T asinh(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C++ code
    return std::asinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float asinhf(float arg) {
    return asinh(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double asinhl(long double arg) {
    return asinh(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double asinhf(T arg) {
    return asinh(static_cast<double>(arg));
  }

}  // namespace xtd
