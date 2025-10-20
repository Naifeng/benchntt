# AMD and Intel compiler configurations
INTEL_CC = icx
INTEL_CFLAGS = -march=skylake-avx512 -O3 -w
AMD_CC = clang
AMD_CFLAGS = -march=znver4 -O3 -w

# default settings using GCC
CC = gcc
CFLAGS = -march=native -O3 -w

SRC_DIR = src
TARGET_DIR = bin

# target executables
BLAS_TARGETS = blas_scalar blas_avx2 blas_avx512
NTT_TARGETS = ntt_scalar ntt_avx2 ntt_avx512

MAIN_TARGETS = blas ntt

PLATFORM_TARGETS = $(foreach target,$(MAIN_TARGETS),$(target)-intel $(target)-amd)
.PHONY: all clean $(MAIN_TARGETS) $(PLATFORM_TARGETS)

all: blas ntt

$(TARGET_DIR):
	mkdir -p $(TARGET_DIR)

$(TARGET_DIR)/%: $(SRC_DIR)/%.c $(SRC_DIR)/config.c | $(TARGET_DIR)
	$(CC) $(CFLAGS) -o $@ $^

define compiler-settings
$(1)-intel: override CC := $(INTEL_CC)
$(1)-intel: override CFLAGS := $(INTEL_CFLAGS)
$(1)-intel:
	@$(MAKE) --no-print-directory $(1) CC="$$(CC)" CFLAGS="$$(CFLAGS)"

$(1)-amd: override CC := $(AMD_CC)
$(1)-amd: override CFLAGS := $(AMD_CFLAGS)
$(1)-amd:
	@$(MAKE) --no-print-directory $(1) CC="$$(CC)" CFLAGS="$$(CFLAGS)"
endef

$(foreach target,$(MAIN_TARGETS),$(eval $(call compiler-settings,$(target))))

blas: $(addprefix $(TARGET_DIR)/, $(BLAS_TARGETS))
	@echo ""
	@echo "Running BLAS operation benchmarks..."
	@echo ""
	@cd $(TARGET_DIR) && for target in $(BLAS_TARGETS); do \
		./$$target; \
	done

ntt: $(addprefix $(TARGET_DIR)/, $(NTT_TARGETS))
	@echo ""
	@echo "Running NTT benchmarks..."
	@echo ""
	@chmod +x benchmark/run_ntt.sh
	@benchmark/run_ntt.sh

clean:
	rm -rf $(TARGET_DIR)