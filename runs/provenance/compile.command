taskset -c 0-7 /usr/local/cuda-13.1/bin/nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo -Xptxas=-v soft_voronoi_bench.next.cu -o build/soft_voronoi_bench.next
