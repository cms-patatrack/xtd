
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr float acos(T arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C++ code
    return std::acos(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float acosf(float arg) {
    return acos(arg);
  }

  XTD_DEVICE_FUNCTION
  inline constexpr long double acosl(long double arg) {
    return acos(arg);
  }

  template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
  XTD_DEVICE_FUNCTION
  inline constexpr double acos(T arg) {
    return acos(static_cast<double>(arg));
  }

}  // namespace xtd
