# Copyright 2025 European Organization for Nuclear Research (CERN)
# Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
# SPDX-License-Identifier: MPL-2.0

.DELETE_ON_ERROR:

.PHONY: all
all: run

.PHONY: build
build:

.PHONY: run
run: build

.PHONY: clean
clean:

.PHONY: distclean
distclean: clean
	rm -rf test/external

# makefile tricks
define newline


endef

space := $() $()
comma := $(),$()

# gcc
CXX := g++
GCC_TOOLCHAIN := $(abspath $(dir $(shell which $(CXX)))/..)
GCC_TARGET    := $(shell $(CXX) -dumpmachine)
# Catch2 needs -Wno-unused-variable
HOST_CXXFLAGS := -O2 -fPIC -pthread -march=native -Wall -Wextra -Werror -Wfatal-errors -Wno-unused-variable

# Compiler flags supported by GCC but not by the LLVM-based compilers (clang, hipcc, icpx, etc.)
LLVM_UNSUPPORTED_CXXFLAGS := --param vect-max-version-for-alias-checks=50 -Werror=format-contains-nul -Wno-non-template-friend -Werror=return-local-addr -Werror=unused-but-set-variable

CXXFLAGS := -std=c++20 $(HOST_CXXFLAGS) -g
LDFLAGS := -O2 -fPIC -pthread -Wl,-E -lstdc++fs -ldl

# CUDA
CUDA_BASE := /usr/local/cuda
# default to the list of all supported major architectures
CUDA_ARCH := 50 60 70 80 90 100 120
ifeq ($(wildcard $(CUDA_BASE)),)
  # CUDA platform not found
  $(info Cannot find an NVIDIA CUDA installation at $(CUDA_BASE), the CUDA tests will not be built.)
  CUDA_BASE :=
else
# CUDA platform at $(CUDA_BASE)
  CUDA_LIBDIR := $(CUDA_BASE)/lib64
  CUDA_DEPS := $(CUDA_LIBDIR)/libcudart.so
  # detect the architecture of the NVIDIA GPUs in the system
  CUDA_DETECTED_ARCH := $(shell $(CUDA_BASE)/bin/__nvcc_device_query 2> /dev/null)
  ifneq ($(CUDA_DETECTED_ARCH),)
    CUDA_ARCH := $(CUDA_DETECTED_ARCH)
  endif
  CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
  CUDA_LDFLAGS := -L$(CUDA_LIBDIR) -lcudart -lcudadevrt
  CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
  define CUFLAGS_template
    $(2)NVCC_FLAGS := $$(foreach ARCH,$(1),-gencode arch=compute_$$(ARCH),code=[sm_$$(ARCH),compute_$$(ARCH)]) -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --threads $$(words $(1)) --cudart=shared
    $(2)NVCC_COMMON := -std=c++20 -O3 -g $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
    $(2)CUDA_CUFLAGS := $$($(2)NVCC_COMMON)
  endef
  $(eval $(call CUFLAGS_template,$(CUDA_ARCH),))
  NVCC_COMMON := -std=c++20 -O3 -g $(NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
  CUDA_CUFLAGS := $(NVCC_COMMON)

  # Make sure the tests can find the CUDA libraries
  export LD_LIBRARY_PATH := $(CUDA_LIBDIR):$(LD_LIBRARY_PATH)
endif

# ROCm
ROCM_BASE := /opt/rocm
# default to the list of all supported architectures
ROCM_ARCH := gfx900 gfx906 gfx908 gfx90a gfx942 gfx1030 gfx1100 gfx1101 gfx1200 gfx1201
ifeq ($(wildcard $(ROCM_BASE)),)
  # ROCm platform not found
  $(info Cannot find an AMD ROCm installation at $(ROCM_BASE), the ROCm tests will not be built.)
  ROCM_BASE :=
else
  # ROCm platform at $(ROCM_BASE)
  ROCM_LIBDIR := $(ROCM_BASE)/lib
  ROCM_DEPS := $(ROCM_LIBDIR)/libamdhip64.so
  # detect the architecture of the AMD GPUs in the system
  ROCM_DETECTED_ARCH := $(shell $(ROCM_BASE)/lib/llvm/bin/amdgpu-arch 2> /dev/null | sort -u)
  ifneq ($(ROCM_DETECTED_ARCH),)
    ROCM_ARCH := $(ROCM_DETECTED_ARCH)
  endif
  ROCM_HIPCC := $(ROCM_BASE)/bin/hipcc
  HIPCC_CXXFLAGS := -fno-gpu-rdc $(foreach ARCH,$(ROCM_ARCH),--offload-arch=$(ARCH)) $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN) -I$(ROCM_BASE)/include/hip -Wno-unused-result
  HIPCC_LDFLAGS := $(LDFLAGS) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN)

  # Make sure the tests can find the ROCm libraries
  export LD_LIBRARY_PATH := $(ROCM_LIBDIR):$(LD_LIBRARY_PATH)
endif

# Intel oneAPI
# look for the oneAPI installation in
#   - the directory specified by ONEAPI_BASE, if set; for example calling:
#       make ONEAPI_BASE=/opt/intel/oneapi
#   - the directory specified by ONEAPI_ROOT, if set; this is usually set by
#       . /opt/intel/oneapi/setvars.sh
#   - /opt/intel/oneapi; this is the default value

# if ONEAPI_BASE is set and not empty, check if it points to a valid oneAPI installation
ifneq ($(ONEAPI_BASE),)
  ifneq ($(wildcard $(ONEAPI_BASE)/setvars.sh),)
    # found Intel oneAPI at $(ONEAPI_BASE)
    $(info Found an Intel oneAPI installation at $(ONEAPI_BASE))
  else
    # Intel oneAPI not found at $(ONEAPI_BASE), set ONEAPI_BASE to an empty value
    $(info Cannot find an Intel oneAPI installation at $(ONEAPI_BASE), the oneAPI tests will not be built.)
    override ONEAPI_BASE :=
  endif
endif

# if ONEAPI_BASE is not set (not just empty) and ONEAPI_ROOT is set and not empty, check if it points to a valid oneAPI installation
ifeq ($(origin ONEAPI_BASE), undefined)
  ifdef ONEAPI_ROOT
    ifneq ($(wildcard $(ONEAPI_ROOT)/setvars.sh),)
      # found Intel oneAPI at $(ONEAPI_ROOT)
      ONEAPI_BASE := $(ONEAPI_ROOT)
      $(info Found an Intel oneAPI installation at $(ONEAPI_BASE))
    endif
  endif
endif

# if ONEAPI_BASE is not set (not just empty), look for a valid oneAPI installation at /opt/intel/oneapi
ifeq ($(origin ONEAPI_BASE), undefined)
  ifneq ($(wildcard /opt/intel/oneapi/setvars.sh),)
    # found Intel oneAPI at /opt/intel/oneapi
    ONEAPI_BASE := /opt/intel/oneapi
    $(info Found an Intel oneAPI installation at $(ONEAPI_BASE))
  endif
endif

# if ONEAPI_BASE is set and not empty, check if the environment has been setup
ifneq ($(ONEAPI_BASE),)
  ifeq ($(shell which sycl-ls 2> /dev/null),)
    # Intel oneAPI environment not setup
    $(info The Intel oneAPI environment has not been setup. Please run:$(newline)    . $(ONEAPI_BASE)/setvars.sh$(newline)before running make.$(newline)Alternatively, disable oneAPI detection setting ONEAPI_BASE to an empty value, for eample:$(newline)    make ONEAPI_BASE=$(newline))
    override ONEAPI_BASE :=
  endif
endif
ifneq ($(ONEAPI_BASE),)
  SYCL_BASE         := $(ONEAPI_BASE)/compiler/latest
  SYCL_LIBDIR       := $(SYCL_BASE)/lib
  SYCL_CXX          := $(SYCL_BASE)/bin/icpx
  SYCL_CXXFLAGS     := $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS))
  SYCL_LDFLAGS      :=
  SYCL_TARGETS      := spir64_x86_64 spir64
  ifdef CUDA_BASE
    ifneq ($(wildcard $(SYCL_LIBDIR)/libur_adapter_cuda.so),)
      # Note: as of oneAPI 2025.2 icpx can target a single CUDA architecture.
      # If the syntax
      #   -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_80 -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_90
      # is used, the backend compiler is invoked only once for the first architecture (sm_80), and the others are ignored.
      # If the sytnax
      #   -fsycl-targets=nvidia_gpu_sm_80,nvidia_gpu_sm_90
      # is used, the backend compiler is invoked multiple times, but always for the first architecture (sm_80).
      # See https://github.com/intel/llvm/issues/16424 for some examples.
      #SYCL_TARGETS    += $(foreach SM,$(CUDA_ARCH),nvidia_gpu_sm_$(SM))
      # To avoid inconsistencies, we build the tests only for the first CUDA architecture.
      SYCL_TARGETS    += nvidia_gpu_sm_$(firstword $(CUDA_ARCH))
      SYCL_CUDA_FLAGS := --cuda-path=$(CUDA_BASE)
    endif
  endif
  ifdef ROCM_BASE
    ifneq ($(wildcard $(SYCL_LIBDIR)/libur_adapter_hip.so),)
      # Note: the same issue described above for NVIDIA GPUs applies to AMD GPUs
      #SYCL_TARGETS    += $(foreach GFX,$(ROCM_ARCH),amd_gpu_$(GFX))
      # To avoid inconsistencies, we build the tests only for the first ROCm architecture.
      SYCL_TARGETS    += amd_gpu_$(firstword $(ROCM_ARCH))
      SYCL_ROCM_FLAGS := --rocm-path=$(ROCM_BASE)
    endif
  endif
  SYCL_FLAGS        := -fsycl -fsycl-targets=$(subst $(space),$(comma),$(SYCL_TARGETS)) $(SYCL_CUDA_FLAGS) $(SYCL_ROCM_FLAGS) -fp-model=precise

  # Check for Intel GPU existence
  #SYCL_LS := $(shell mktemp)
  #$(shell which sycl-ls > /dev/null 2>&1 && sycl-ls --verbose 2> /dev/null > $(SYCL_LS))
  #SYCL_GPU_TARGET := -fsycl-targets=$(shell echo $$(cat $(SYCL_LS) | grep Architecture | cut -d: -f2 | sort -u | grep intel_gpu) | tr ' ' ',')
  #$(shell rm -f $(SYCL_LS))
  #undefine SYCL_LS

  # Enable double precision floating point emulation for Intel GPUs
  # See https://github.com/intel/compute-runtime/blob/master/opencl/doc/FAQ.md#feature-double-precision-emulation-fp64
  export IGC_EnableDPEmulation := 1
  export OverrideDefaultFP64Settings := 1
endif

# xtd
XTD_BASE := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/..)

# external Catch2 library
CATCH2_INCLUDE := $(XTD_BASE)/test/external/catch2/include/catch.hpp

.PHONY: external_catch2
external_catch2: $(CATCH2_INCLUDE)

$(CATCH2_INCLUDE):
	mkdir -p $(dir $@)
	wget https://github.com/catchorg/Catch2/releases/download/v2.13.10/catch.hpp -O $@

LIB_INCLUDE := -I$(XTD_BASE)/include -I$(XTD_BASE)/test -I$(dir $(CATCH2_INCLUDE))

# xtd tests
SUBDIRS := $(wildcard $(XTD_BASE)/test/*/)
TARGETS_ALL := $(filter-out common, $(filter-out external, $(notdir $(patsubst %/,%,$(SUBDIRS)))))

define TEST_template
build: build_$(1)

.PHONY: build_$(1)
build_$(1): prereq_$(1)

.PHONY: prereq_$(1)
prereq_$(1): external_catch2 $(1)/bin

run: run_$(1)

.PHONY: run_$(1)
run_$(1):

clean: clean_$(1)

.PHONY: clean_$(1)
clean_$(1):
	rm -rf $(1)/bin

$(1)/bin:
	mkdir -p $(1)/bin

# CPU tests
build_$(1): build_$(1)_cpu

.PHONY: build_$(1)_cpu
build_$(1)_cpu: $(1)/bin/$(1)_t_cc

$(1)/bin/$(1)_t_cc: $(1)/$(1)_t.cc | prereq_$(1)
	$(CXX) $(CXXFLAGS) $(LIB_INCLUDE) $$< -o $$@

run_$(1): run_$(1)_cpu

.PHONY: run_$(1)_cpu
run_$(1)_cpu: $(1)/bin/$(1)_t_cc
	$$<

# CUDA tests
ifdef CUDA_BASE
build_$(1): build_$(1)_cuda

.PHONY: build_$(1)_cuda
build_$(1)_cuda: $(1)/bin/$(1)_t_cuda

$(1)/bin/$(1)_t_cuda: $(1)/$(1)_t.cu | prereq_$(1)
	$(CUDA_NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) $(CUDA_CUFLAGS) $(LIB_INCLUDE) $$< -o $$@

run_$(1): run_$(1)_cuda

.PHONY: run_$(1)_cuda
run_$(1)_cuda: $(1)/bin/$(1)_t_cuda
	$$<
endif

# HIP/ROCm tests
ifdef ROCM_BASE
build_$(1): build_$(1)_hip

.PHONY: build_$(1)_hip
build_$(1)_hip: $(1)/bin/$(1)_t_hip

$(1)/bin/$(1)_t_hip: $(1)/$(1)_t.hip.cc | prereq_$(1)
	$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(HIPCC_LDFLAGS) $(LIB_INCLUDE) $$< -o $$@

run_$(1): run_$(1)_hip

.PHONY: run_$(1)_hip
run_$(1)_hip: $(1)/bin/$(1)_t_hip
	$$<
endif

# oneAPI tests
ifdef ONEAPI_BASE
build_$(1): build_$(1)_sycl

.PHONY: build_$(1)_sycl
build_$(1)_sycl: $(1)/bin/$(1)_t_sycl

$(1)/bin/$(1)_t_sycl: $(1)/$(1)_t.sycl.cc | prereq_$(1)
	$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(SYCL_FLAGS) $(LIB_INCLUDE) $$< -o $$@

run_$(1): run_$(1)_sycl

.PHONY: run_$(1)_sycl
run_$(1)_sycl: $(1)/bin/$(1)_t_sycl
	$$<
endif

endef

$(foreach target,$(TARGETS_ALL),$(eval $(call TEST_template,$(target))))
