# shielded-worker -- the C++/CUDA worker.
#
# Compiled with clang++ as the CUDA compiler, against the pip-wheel toolkit in
# CUDA_HOME: nvcc 12.6 refuses this box's gcc 16 outright, and clang handles
# both the device code and the modern libstdc++. Carries libcudart.so.12 beside
# the binary, so at run time it depends only on libcuda, which the driver provides.
#
# Links the SAME shielded-field.c the TEE links, so the legacy wq/wd install path
# encodes weights with the identical routine. fp contraction is OFF on purpose:
# an FMA would round differently from the TEE and unmasking would return noise.

CUDA_HOME ?= $(HOME)/.cache/sd-gpu-repro/cuda-home
CUDACXX   ?= clang++
FIELD_DIR := ../../wasm/ggml-shielded
ARCHS     ?= --cuda-gpu-arch=sm_70 --cuda-gpu-arch=sm_80 --cuda-gpu-arch=sm_86 --cuda-gpu-arch=sm_89 --cuda-gpu-arch=sm_90

all: shielded-worker

shielded-field.o: $(FIELD_DIR)/shielded-field.c $(FIELD_DIR)/shielded-field.h
	$(CC) -O2 -ffp-contract=off -fPIC -I$(FIELD_DIR) -c $< -o $@

worker.o: worker.cu $(FIELD_DIR)/shielded-field.h Makefile
	$(CUDACXX) -O3 -std=c++17 -x cuda --cuda-path=$(CUDA_HOME) --no-cuda-version-check $(ARCHS) \
	  -ffp-contract=off -I$(FIELD_DIR) -c worker.cu -o $@

# libcudart.so.12 is copied beside the binary (rpath $ORIGIN first) so the
# deployed worker carries its own runtime; the toolkit dir is the dev fallback.
# Linked to a temporary name and moved into place ATOMICALLY. On a metal host
# the launcher runs the worker straight from this path; linking over the live
# binary rewrites the pages a running process is executing and it dies in the
# middle of an exchange (2026-08-26: SIGSEGV in a symbol from the NEW build,
# the tenant's link gone, the app on the int64 path at 6 tok/s until relaunch).
# A rename keeps the old inode alive for the old process; the launcher picks
# up the new binary on its next (re)start.
shielded-worker: worker.o shielded-field.o
	cp -f $(CUDA_HOME)/lib64/libcudart.so.12 libcudart.so.12.tmp && mv -f libcudart.so.12.tmp libcudart.so.12
	$(CUDACXX) -o $@.tmp worker.o shielded-field.o -L$(CUDA_HOME)/lib64 -lcudart -ldl -lrt -lpthread \
	  -Wl,-rpath,'$$ORIGIN' -Wl,-rpath,$(CUDA_HOME)/lib64
	mv -f $@.tmp $@

clean:
	rm -f shielded-worker worker.o shielded-field.o libcudart.so.12
.PHONY: all clean
