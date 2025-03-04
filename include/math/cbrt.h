
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) &&                   \
    !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T cbrt(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::cbrt(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::cbrt(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::cbrt(x);
#else
  // standard C++ code
  return std::cbrt(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float cbrtf(float x) { return cbrt(x); }

XTD_DEVICE_FUNCTION inline constexpr long double cbrtl(long double x) {
  return cbrt(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double cbrt(T x) {
  return cbrt(static_cast<double>(x));
}

} // namespace xtd
