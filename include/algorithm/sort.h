
#pragma once

#include "internal/defines.h"

#if defined(XTD_CUDA_BACKEND) || defined(XTD_HIP_BACKEND)
#include <thrust/sort.h>
#elif defined(XTD_SYCL_BACKEND)
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#else
#include <algorithm>
#endif

namespace xtd {

  template <typename RandomAccessIterator>
  XTD_HOST_FUNCTION inline constexpr void sort(RandomAccessIterator first,
                                               RandomAccessIterator last) {
#if defined(XTD_CUDA_BACKEND)
    thrust::sort(thrust::device, first, last);
#elif defined(XTD_HIP_BACKEND)
    rocthrust::sort(thrustd::hip::par, first, last);
#elif defined(XTD_SYCL_BACKEND)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last);
#else
    std::sort(first, last);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator>
  XTD_HOST_FUNCTION inline constexpr void sort(ExecutionPolicy&& policy,
                                               RandomAccessIterator first,
                                               RandomAccessIterator last) {
#if defined(XTD_CUDA_BACKEND)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_HIP_BACKEND)
    rocthrust::sort(std::forward<ExecutionPolicy>(policy), first, last);
#elif defined(XTD_SYCL_BACKEND)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last);
#endif
  }

  template <typename RandomAccessIterator, typename Compare>
  XTD_HOST_FUNCTION inline constexpr void sort(RandomAccessIterator first,
                                               RandomAccessIterator last,
                                               Compare comp) {
#if defined(XTD_CUDA_BACKEND)
    thrust::sort(thrust::device, first, last, comp);
#elif defined(XTD_HIP_BACKEND)
    rocthrust::sort(thrust::hip::par, first, last, comp);
#elif defined(XTD_SYCL_BACKEND)
    oneapi::dpl::sort(oneapi::dpl::execution::dpcpp_default, first, last, comp);
#else
    std::sort(first, last, comp);
#endif
  }

  template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
  XTD_HOST_FUNCTION inline constexpr void sort(ExecutionPolicy&& policy,
                                               RandomAccessIterator first,
                                               RandomAccessIterator last,
                                               Compare comp) {
#if defined(XTD_CUDA_BACKEND)
    thrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_HIP_BACKEND)
    rocthrust::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#elif defined(XTD_SYCL_BACKEND)
    oneapi::dpl::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#else
    std::sort(std::forward<ExecutionPolicy>(policy), first, last, comp);
#endif
  }

}  // namespace xtd
