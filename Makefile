# Prefer installed SCIP prefix (include/, lib/). Override with source-tree layout if needed.
SCIP_ROOT ?= $(CURDIR)/artifacts/environment/phase4/scip804_prefix
TORCH_ROOT ?= /data/hanchengcheng/envs/rl4scip/lib/python3.11/site-packages/torch
CXX ?= g++
CXXFLAGS ?= -O2
CXXFLAGS += -std=c++17 -Wall -Wextra -D_GLIBCXX_USE_CXX11_ABI=0
# Prefix layout: include/scip ; source-tree layout: scip/src + build/lib
ifeq ($(wildcard $(SCIP_ROOT)/include/scip/scip.h),)
SCIP_INCLUDES := -I$(SCIP_ROOT)/scip/src -I$(SCIP_ROOT)/build/scip
SCIP_LDFLAGS := -L$(SCIP_ROOT)/build/lib -Wl,-rpath,$(SCIP_ROOT)/build/lib -lscip
else
SCIP_INCLUDES := -I$(SCIP_ROOT)/include
SCIP_LDFLAGS := -L$(SCIP_ROOT)/lib -Wl,-rpath,$(SCIP_ROOT)/lib -lscip
endif
TORCH_INCLUDES := -isystem $(TORCH_ROOT)/include -isystem $(TORCH_ROOT)/include/torch/csrc/api/include
TORCH_LDFLAGS := -L$(TORCH_ROOT)/lib -Wl,-rpath,$(TORCH_ROOT)/lib \
	-ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10
SOURCES := code/scip_tree.cpp src/rl/rl_branchrule.cpp src/rl/scip_feature_extractor.cpp \
	src/rl/model_runner.cpp src/rl/rl_mlp_branchrule.cpp \
	src/rl/scip_graph_feature_extractor.cpp src/rl/gcnn_model_runner.cpp \
	src/rl/rl_gcnn_branchrule.cpp src/rl/prim_bias.cpp
HEADERS := src/rl/rl_branchrule.hpp src/rl/scip_feature_extractor.hpp \
	src/rl/model_runner.hpp src/rl/rl_mlp_branchrule.hpp \
	src/rl/scip_graph_feature_extractor.hpp src/rl/gcnn_model_runner.hpp \
	src/rl/rl_gcnn_branchrule.hpp src/rl/prim_bias.hpp
OBJECTS := $(patsubst %.cpp,build/%.o,$(SOURCES))

.PHONY: all clean test-custom-branching model-runner-parity gcnn-model-runner-parity sb_native_probe

all: build/scip_tree

sb_native_probe: build/sb_native_probe

build/sb_native_probe: tools/sb_native_probe.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) $< $(SCIP_INCLUDES) $(SCIP_LDFLAGS) -o $@

build/scip_tree: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(SCIP_LDFLAGS) $(TORCH_LDFLAGS) -o $@

build/%.o: %.cpp $(HEADERS)
	mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Isrc $(SCIP_INCLUDES) $(TORCH_INCLUDES) -c $< -o $@

clean:
	rm -f build/scip_tree $(OBJECTS)

test-custom-branching: tests/test_custom_branchrule.cpp src/rl/rl_branchrule.cpp src/rl/scip_feature_extractor.cpp $(HEADERS)
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/test_custom_branchrule.cpp src/rl/rl_branchrule.cpp src/rl/scip_feature_extractor.cpp \
		-Isrc $(SCIP_INCLUDES) $(SCIP_LDFLAGS) -o build/test_custom_branchrule
	./build/test_custom_branchrule

model-runner-parity: tests/model_runner_parity.cpp src/rl/model_runner.cpp src/rl/model_runner.hpp src/rl/scip_feature_extractor.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/model_runner_parity.cpp src/rl/model_runner.cpp \
		-Isrc $(SCIP_INCLUDES) $(TORCH_INCLUDES) $(TORCH_LDFLAGS) -o build/model_runner_parity

gcnn-model-runner-parity: tests/gcnn_model_runner_parity.cpp src/rl/gcnn_model_runner.cpp src/rl/gcnn_model_runner.hpp src/rl/scip_graph_feature_extractor.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) tests/gcnn_model_runner_parity.cpp src/rl/gcnn_model_runner.cpp \
		-Isrc $(SCIP_INCLUDES) $(TORCH_INCLUDES) $(TORCH_LDFLAGS) -o build/gcnn_model_runner_parity
