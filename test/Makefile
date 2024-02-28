.PHONY: all build run clean

all: build run

# gcc
CXX := g++
GCC_TOOLCHAIN := $(abspath $(dir $(shell which $(CXX)))/..)
GCC_TARGET    := $(shell $(CXX) -dumpmachine)
# Catch2 needs -Wno-unused-variable
HOST_CXXFLAGS := -O2 -fPIC -pthread -march=native -Wall -Wextra -Werror -Wfatal-errors -Wno-unused-variable

# Compiler flags supported by GCC but not by the LLVM-based compilers (clang, hipcc, icpx, etc.)
LLVM_UNSUPPORTED_CXXFLAGS := --param vect-max-version-for-alias-checks=50 -Werror=format-contains-nul -Wno-non-template-friend -Werror=return-local-addr -Werror=unused-but-set-variable

CXXFLAGS := -std=c++17 $(HOST_CXXFLAGS) -g
LDFLAGS := -O2 -fPIC -pthread -Wl,-E -lstdc++fs -ldl

# CUDA
CUDA_BASE := /usr/local/cuda
ifeq ($(wildcard $(CUDA_BASE)),)
  # CUDA platform not found
  $(warning Cannot find an NVIDIA CUDA installation at $(CUDA_BASE))
  CUDA_BASE :=
else
# CUDA platform at $(CUDA_BASE)
  CUDA_LIBDIR := $(CUDA_BASE)/lib64
  CUDA_DEPS := $(CUDA_LIBDIR)/libcudart.so
  CUDA_ARCH := 60 70 80
  CUDA_CXXFLAGS := -I$(CUDA_BASE)/include
  CUDA_LDFLAGS := -L$(CUDA_LIBDIR) -lcudart -lcudadevrt
  CUDA_NVCC := $(CUDA_BASE)/bin/nvcc
  define CUFLAGS_template
    $(2)NVCC_FLAGS := $$(foreach ARCH,$(1),-gencode arch=compute_$$(ARCH),code=[sm_$$(ARCH),compute_$$(ARCH)]) -Wno-deprecated-gpu-targets -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr --expt-extended-lambda --generate-line-info --source-in-ptx --display-error-number --threads $$(words $(1)) --cudart=shared
    $(2)NVCC_COMMON := -std=c++17 -O3 -g $$($(2)NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
    $(2)CUDA_CUFLAGS := $$($(2)NVCC_COMMON)
  endef
  $(eval $(call CUFLAGS_template,$(CUDA_ARCH),))
  NVCC_COMMON := -std=c++17 -O3 -g $(NVCC_FLAGS) -ccbin $(CXX) --compiler-options '$(HOST_CXXFLAGS)'
  CUDA_CUFLAGS := $(NVCC_COMMON)
endif

# ROCm
ROCM_BASE := /opt/rocm
ifeq ($(wildcard $(ROCM_BASE)),)
  # ROCm platform not found
  $(warning Cannot find an AMD ROCm installation at $(ROCM_BASE))
  ROCM_BASE :=
else
  # ROCm platform at $(ROCM_BASE)
  ROCM_LIBDIR := $(ROCM_BASE)/lib
  ROCM_DEPS := $(ROCM_LIBDIR)/libamdhip64.so
  ROCM_ARCH := gfx900 gfx90a gfx1030
  ROCM_HIPCC := $(ROCM_BASE)/bin/hipcc
  HIPCC_CXXFLAGS := -fno-gpu-rdc $(foreach ARCH,$(ROCM_ARCH),--offload-arch=$(ARCH)) $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN) -I$(ROCM_BASE)/include/hip -Wno-unused-result
  HIPCC_LDFLAGS := $(LDFLAGS) --target=$(GCC_TARGET) --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

# oneAPI
ONEAPI_BASE := /opt/intel/oneapi
ifeq ($(wildcard $(ONEAPI_BASE)),)
  # Intel oneAPI not available
  $(warning Cannot find an Intel oneAPI installation at $(ONEAPI_BASE))
  ONEAPI_BASE :=
else
  SYCL_BASE         := $(ONEAPI_BASE)/compiler/latest
  SYCL_LIBDIR       := $(SYCL_BASE)/lib
  SYCL_CXX          := $(SYCL_BASE)/bin/icpx
  SYCL_CPU_TARGET   := -fsycl-targets=spir64_x86_64
  SYCL_FLAGS        := -fsycl -fp-model=precise
  SYCL_CXXFLAGS     := $(filter-out $(LLVM_UNSUPPORTED_CXXFLAGS),$(CXXFLAGS)) $(SYCL_FLAGS) -Wno-unused-variable
  SYCL_LDFLAGS      :=

  # Check for Intel GPU existence
  SYCL_LS := $(shell mktemp)
  $(shell sycl-ls > $(SYCL_LS))
  INTEL_GPU_EXISTS := $(shell cat $(SYCL_LS) | grep -c ext_oneapi_level_zero)
  # Set AOT_INTEL_TARGETS based on the conditions
  ifneq ($(INTEL_GPU_EXISTS),0)
      ifneq ($(shell cat $(SYCL_LS) | grep -c 'GPU Flex'),0)
          AOT_INTEL_TARGETS := -fsycl-targets=intel_gpu_acm_g10
      else ifneq ($(shell cat $(SYCL_LS) | grep -c 'GPU Max'),0)
          AOT_INTEL_TARGETS := -fsycl-targets=intel_gpu_pvc
      else
          # rely on JIT
          AOT_INTEL_TARGETS := -fsycl-targets=spir64
      endif
  else
      INTEL_GPU_EXISTS :=
      AOT_INTEL_TARGETS :=
  endif
  $(shell rm -f $(SYCL_LS))
  undefine SYCL_LS
endif

# xtd
XTD_BASE := $(realpath $(dir $(realpath $(lastword $(MAKEFILE_LIST))))/..)

# external Catch2 library
CATCH2_INCLUDE := $(XTD_BASE)/test/external/catch2/include/catch.hpp

external_catch2: $(CATCH2_INCLUDE)

$(CATCH2_INCLUDE):
	mkdir -p $(dir $@)
	wget https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp -O $@

LIB_INCLUDE := -I$(XTD_BASE)/include -I$(XTD_BASE)/test -I$(dir $(CATCH2_INCLUDE))

# xtd tests
SUBDIRS := $(wildcard $(XTD_BASE)/test/*/)
TARGETS_ALL := $(filter-out common, $(filter-out external, $(notdir $(patsubst %/,%,$(SUBDIRS)))))

define TEST_template
$(1)/bin:
	mkdir -p $(1)/bin

$(1)Test: external_catch2 $(1)/bin/$(1)_t_cc $(1)/bin/$(1)_t_cuda $(1)/bin/$(1)_t_hip $(1)/bin/$(1)_t_cpusycl $(1)/bin/$(1)_t_gpusycl

$(1)/bin/$(1)_t_cc: $(1)/$(1)_t.cc | $(1)/bin
	$(CXX) $(CXXFLAGS) $(LIB_INCLUDE) $$< -o $$@

$(1)/bin/$(1)_t_cuda: $(1)/$(1)_t.cu | $(1)/bin
	@if [ -z "$(CUDA_BASE)" ]; then \
		echo "Error: CUDA_BASE not set. Skipping $@"; \
	else \
		echo "$(CUDA_NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) $(CUDA_CUFLAGS) $(LIB_INCLUDE) $$< -o $$@"; \
		$(CUDA_NVCC) $(CUDA_CXXFLAGS) $(CUDA_LDFLAGS) $(CUDA_CUFLAGS) $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_hip: $(1)/$(1)_t.hip.cc | $(1)/bin
	@if [ -z "$(ROCM_BASE)" ]; then \
		echo "Error: ROCM_BASE not set. Skipping $@"; \
	else \
		echo "$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(HIPCC_LDFLAGS) $(LIB_INCLUDE) $$< -o $$@"; \
		$(ROCM_HIPCC) $(HIPCC_CXXFLAGS) $(HIPCC_LDFLAGS) $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_cpusycl: $(1)/$(1)_t.sycl.cc | $(1)/bin
	@if [ -z "$(ONEAPI_BASE)" ]; then \
		echo "Error: ONEAPI_BASE not set. Skipping $@"; \
	else \
		echo "$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(SYCL_CPU_TARGET) -DONEAPI_CPU $(LIB_INCLUDE) $$< -o $$@"; \
		$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(SYCL_CPU_TARGET) -DONEAPI_CPU $(LIB_INCLUDE) $$< -o $$@; \
	fi

$(1)/bin/$(1)_t_gpusycl: $(1)/$(1)_t.sycl.cc | $(1)/bin
	@if [ -z "$(ONEAPI_BASE)" ]; then \
		echo "Error: ONEAPI_BASE not set. Skipping $@"; \
	else \
		if [ "$(INTEL_GPU_EXISTS)" ]; then \
			echo "$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_INTEL_TARGETS) $(LIB_INCLUDE) $$< -o $$@"; \
			$(SYCL_CXX) $(SYCL_CXXFLAGS) $(SYCL_LDFLAGS) $(AOT_INTEL_TARGETS) $(LIB_INCLUDE) $$< -o $$@; \
		else \
    		echo "Error: Intel GPU not found. Skipping $@"; \
    	fi \
	fi

# List of test executables
$(1)_BIN := $(XTD_BASE)/test/$(1)/bin

# Add targets
TEST_EXECUTABLES := $(1)/bin/$(1)_t_cc
ifdef $(CUDA_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_cuda
endif
ifdef $(ROCM_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_hip
endif
ifdef $(ONEAPI_BASE)
  TEST_EXECUTABLES += $(1)/bin/$(1)_t_cpusycl
  ifdef $(INTEL_GPU_EXISTS)
    TEST_EXECUTABLES += $(1)/bin/$(1)_t_gpusycl
  endif
endif

run_$(1)Test: $(TEST_EXECUTABLES)
	@find $$($(1)_BIN) -maxdepth 1 -type f -exec echo "Running {}" \; -exec {} \; -exec echo \;

clean_$(1)Test:
	rm -rf $(1)/bin
endef

$(foreach target,$(TARGETS_ALL),$(eval $(call TEST_template,$(target))))

build: $(foreach target,$(TARGETS_ALL), $(target)Test)

run: $(foreach target,$(TARGETS_ALL), run_$(target)Test)

clean: $(foreach target,$(TARGETS_ALL), clean_$(target)Test)
