# shielded ggml backend -- the TEE-side half that lets an ordinary ggml graph
# offload its linear ops to an untrusted GPU.
#
# Built standalone here so the field/wire/tee layers can be tested without the
# engine; the backend object is what gets linked into the ELL build.

CC      ?= cc
CFLAGS  ?= -O2 -Wall -Wextra -fPIC
LDLIBS  += -lm

CORE_SRC := shielded-field.c shielded-wire.c shielded-tee.c shielded-pads.c tweetnacl.c
CORE_OBJ := $(CORE_SRC:.c=.o) shielded-simd-avx512.o shielded-simd-generic.o
LDLIBS   += -lpthread

# The field encoder must not be contracted into FMAs: the worker runs the same
# source and both sides have to round identically.
shielded-field.o: CFLAGS += -ffp-contract=off

# The ggml backend needs ggml's headers and libraries. They are NOT a build
# dependency of the core above, on purpose: the field/wire/tee layers are what
# the reference can be checked against, and they must stay testable on a machine
# with no engine checkout at all (CI is one).
GGML_SRC ?= $(HOME)/Projects/llama.cpp
GGML_LIB ?= $(HOME)/Projects/llamacpp-lib
GGML_INC := -I$(GGML_SRC)/ggml/include -I$(GGML_SRC)/ggml/src
CXX      ?= c++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -fPIC

all: shielded-probe field-selftest prepare-selftest half-selftest refill-selftest dealt-selftest

# Built only where an engine checkout exists; `make ggml` opts in.
ggml: ggml-test

# Whole-model runner; needs libllama as well as ggml.
run: shielded-run

# The calibrator: any q8_0 GGUF libllama can prefill -> its .calib for the C
# backend. Needs libllama too. Against an ELL engine drop (no source checkout)
# point the include and lib variables at it:
#   make calib GGML_LIB=$ELL/lib LLAMA_INC=-I$ELL_HDR/include GGML_INC=-I$ELL_HDR/ggml-include
#   ./shielded-calib --backend $ELL/lib/libggml-cpu.so model.gguf model.calib
calib: shielded-calib
LLAMA_INC ?= -I$(GGML_SRC)/include

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# The hot loops, twice: the AVX-512 VNNI build and the one that runs anywhere.
# shielded-tee.c checks them against each other before trusting the fast one.
shielded-simd-avx512.o: shielded-simd.c shielded-simd.h shielded-field.h
	$(CC) $(CFLAGS) -O3 -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni -DSH_SIMD_AVX512 -c $< -o $@
shielded-simd-generic.o: shielded-simd.c shielded-simd.h shielded-field.h
	$(CC) $(CFLAGS) -O3 -c $< -o $@

shielded-probe: shielded-probe.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

field-selftest: field-selftest.c shielded-field.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

prepare-selftest: prepare-selftest.c shielded-field.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

half-selftest: half-selftest.c shielded-field.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# The refill kernels against a scalar exact product (exit 77 = no AVX-512 VNNI here).
refill-selftest: refill-selftest.c shielded-simd-avx512.o shielded-field.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Dealt pads end to end on synthetic weights: mint, read back against an
# exact oracle, tamper, exhaustion, ledger window, the link's own import.
dealt-selftest: dealt-selftest.c $(CORE_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

ggml-shielded.o: ggml-shielded.cpp ggml-shielded.h shielded-latency.h shielded-tee.h shielded-field.h shielded-simd.h
	$(CXX) $(CXXFLAGS) $(GGML_INC) -c $< -o $@

ggml-test: ggml-test.cpp ggml-shielded.o $(CORE_OBJ)
	$(CXX) $(CXXFLAGS) $(GGML_INC) -o $@ $^ \
	  -L$(GGML_LIB) -lggml -lggml-base -lggml-cpu $(LDLIBS) -Wl,-rpath,$(GGML_LIB)

ggml-shielded-dl.o: ggml-shielded.cpp ggml-shielded.h shielded-latency.h shielded-tee.h shielded-field.h shielded-simd.h
	$(CXX) $(CXXFLAGS) -DGGML_BACKEND_DL -DGGML_BACKEND_SHARED $(GGML_INC) -c $< -o $@

libggml-shielded.so: ggml-shielded-dl.o $(CORE_OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ -L$(GGML_LIB) -lggml-base $(LDLIBS) -Wl,-rpath,$(GGML_LIB)

shielded-run: shielded-run.cpp libggml-shielded.so
	$(CXX) $(CXXFLAGS) -I$(GGML_SRC)/include $(GGML_INC) -o $@ $< \
	  -L$(GGML_LIB) -lllama -lggml -lggml-base -ldl -Wl,-rpath,$(GGML_LIB)

# shielded-field.o is the SAME object the backend links, -ffp-contract=off and
# all: the calibrator's w_fixed must be the runtime's, byte for byte, or the
# exponent it picks is for a product the runtime never forms.
shielded-calib: shielded-calib.cpp shielded-field.o shielded-field.h
	$(CXX) $(CXXFLAGS) $(LLAMA_INC) $(GGML_INC) -o $@ shielded-calib.cpp shielded-field.o \
	  -L$(GGML_LIB) -lllama -lggml -lggml-base $(LDLIBS) -Wl,-rpath,$(GGML_LIB)

# Exchange-count amortiser benches (`make bench`): the same link line as
# shielded-run, backend modules loaded by path at run time (BACKENDS env), so
# cpu / cuda / shielded rows come from identical engine code.
#   bench-batch  ms per decode step of M rows (a verify pass of K=M-1 drafts,
#                or M concurrent users): the batch-width cost curve.
#   bench-spec   real MTP self-drafting speculation through libenclave_llama's
#                own verbs (draft2 / decode_seq_full / rewind / harvest).
bench: bench-batch bench-spec

# The dealer: mint .pads shipments from a model; libllama for the load, the
# backend module for the registration and the mint (shielded/dealer/PLAN.md).
shielded-dealer: shielded-dealer.cpp shielded-pads.o tweetnacl.o shielded-tee.o shielded-field.o shielded-wire.o shielded-simd-avx512.o shielded-simd-generic.o
	$(CXX) $(CXXFLAGS) $(LLAMA_INC) $(GGML_INC) -I../llama-shim -o $@ $^ \
	  -L$(GGML_LIB) -lllama -lggml -lggml-base -ldl $(LDLIBS) -Wl,-rpath,$(GGML_LIB)

bench-batch: bench-batch.cpp
	$(CXX) $(CXXFLAGS) -I$(GGML_SRC)/include $(GGML_INC) -o $@ $< \
	  -L$(GGML_LIB) -lllama -lggml -lggml-base -ldl -Wl,-rpath,$(GGML_LIB)

bench-spec: bench-spec.cpp ../llama-shim/enclave_llama.h
	$(CXX) $(CXXFLAGS) -I$(GGML_SRC)/include $(GGML_INC) -I../llama-shim -o $@ $< \
	  -L$(GGML_LIB) -lenclave_llama -lllama -lggml -lggml-base -ldl -Wl,-rpath,$(GGML_LIB)

clean:
	rm -f $(CORE_OBJ) ggml-shielded-dl.o shielded-probe field-selftest prepare-selftest half-selftest refill-selftest dealt-selftest ggml-shielded.o ggml-test shielded-run shielded-calib libggml-shielded.so bench-batch bench-spec

.PHONY: all clean ggml run calib bench
