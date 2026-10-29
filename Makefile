NVCC ?= nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xptxas=-v

.PHONY: all test clean
all: build/soft_voronoi_bench
build/soft_voronoi_bench: soft_voronoi_bench.cu
	mkdir -p build
	@printf '%s\n' '$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@' > build/compile.command
	@$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@ > build/compile.stdout 2> build/compile.stderr || { cat build/compile.stderr >&2; exit 1; }
	@cat build/compile.stderr >&2
test: all
	./build/soft_voronoi_bench --self-test
clean:
	rm -f build/soft_voronoi_bench
