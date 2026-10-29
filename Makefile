NVCC ?= nvcc
ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -lineinfo -Xptxas=-v

.PHONY: all test clean
all: build/soft_voronoi_bench
build/soft_voronoi_bench: soft_voronoi_bench.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) -arch=$(ARCH) $< -o $@
test: all
	./build/soft_voronoi_bench --self-test
clean:
	rm -f build/soft_voronoi_bench
