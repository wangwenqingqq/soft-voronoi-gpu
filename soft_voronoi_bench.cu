// Exact bounded-string soft Voronoi feasibility benchmark. See docs/core-design.md.
// Build: nvcc -O3 -std=c++17 -arch=sm_120 -lineinfo -Xptxas=-v soft_voronoi_bench.cu -o
// soft_voronoi_bench
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using U64 = unsigned long long;
using U32 = unsigned int;
constexpr int MAXLEN = 256, WORDS = 4, ALPHABET = 256, THREADS = 256, TOPK = 10, MAXKB = 16;
constexpr U64 INVALID = ~U64(0);
#define CUDA(call)                                                                                 \
  do {                                                                                             \
    cudaError_t e = (call);                                                                        \
    if (e != cudaSuccess)                                                                          \
      throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(e));                 \
  } while (0)
#define LAUNCH() CUDA(cudaGetLastError())
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
static void log(const std::string &s) {
  std::time_t t = std::time(nullptr);
  std::cerr << std::put_time(std::gmtime(&t), "%Y-%m-%dT%H:%M:%SZ") << " " << s << std::endl;
}
static std::string quote(const std::string &s) {
  std::string r = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      r += '\\';
      r += char(c);
    } else if (c < 32)
      r += '?';
    else
      r += char(c);
  }
  return r + '"';
}
template <class T> struct Device {
  T *p = nullptr;
  size_t n = 0;
  Device() = default;
  explicit Device(size_t k) { resize(k); }
  ~Device() {
    if (p)
      cudaFree(p);
  }
  Device(const Device &) = delete;
  Device &operator=(const Device &) = delete;
  Device(Device &&x) noexcept : p(x.p), n(x.n) {
    x.p = nullptr;
    x.n = 0;
  }
  Device &operator=(Device &&x) noexcept {
    if (this != &x) {
      if (p)
        cudaFree(p);
      p = x.p;
      n = x.n;
      x.p = nullptr;
      x.n = 0;
    }
    return *this;
  }
  void resize(size_t k) {
    if (p)
      CUDA(cudaFree(p));
    p = nullptr;
    n = k;
    if (k)
      CUDA(cudaMalloc(&p, k * sizeof(T)));
  }
};
struct Events {
  cudaEvent_t e[6];
  Events() {
    for (auto &v : e)
      CUDA(cudaEventCreate(&v));
  }
  ~Events() {
    for (auto v : e)
      cudaEventDestroy(v);
  }
};

// Full-width Myers arithmetic. Addition and shifts propagate across every word;
// padding above the last pattern bit cannot feed lower bits and is irrelevant.
template <int W>
__device__ __forceinline__ int myers(const U64 *masks, int plen, const unsigned char *text,
                                     int stride, int row, int tlen) {
  U64 pv[W], mv[W];
#pragma unroll
  for (int w = 0; w < W; ++w) {
    pv[w] = ~U64(0);
    mv[w] = 0;
  }
  int score = plen;
  const int last = (plen - 1) >> 6;
  const U64 high = U64(1) << ((plen - 1) & 63);
  for (int j = 0; j < tlen; ++j) {
    const U64 *eq = masks + int(text[size_t(j) * stride + row]) * WORDS;
    U64 carry = 0, pcarry = 1, mcarry = 0;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      const U64 e = eq[w], v = pv[w], n = mv[w], xv = e | n, a = e & v;
      const U64 s = a + v, c1 = U64(s < a), s2 = s + carry, c2 = U64(s2 < s);
      carry = c1 | c2;
      const U64 xh = (s2 ^ v) | e;
      U64 ph = n | ~(xh | v), mh = v & xh;
      if (w == last)
        score += int((ph & high) != 0) - int((mh & high) != 0);
      const U64 pn = ph >> 63, mn = mh >> 63;
      ph = (ph << 1) | pcarry;
      mh = (mh << 1) | mcarry;
      pcarry = pn;
      mcarry = mn;
      pv[w] = mh | ~(xv | ph);
      mv[w] = ph & xv;
    }
  }
  return score;
}
__device__ __forceinline__ int distance(const U64 *masks, int plen, const unsigned char *text,
                                        int stride, int row, int tlen) {
  if (!plen)
    return tlen;
  if (plen <= 64)
    return myers<1>(masks, plen, text, stride, row, tlen);
  if (plen <= 128)
    return myers<2>(masks, plen, text, stride, row, tlen);
  if (plen <= 192)
    return myers<3>(masks, plen, text, stride, row, tlen);
  return myers<4>(masks, plen, text, stride, row, tlen);
}
__global__ void make_masks(const unsigned char *strings, const int *lens, int count, U64 *masks) {
  int c = threadIdx.x, p = blockIdx.x;
  if (p >= count)
    return;
  U64 a[WORDS] = {0, 0, 0, 0};
  for (int j = 0; j < lens[p]; ++j)
    if (strings[size_t(j) * count + p] == c)
      a[j >> 6] |= U64(1) << (j & 63);
#pragma unroll
  for (int w = 0; w < WORDS; ++w)
    masks[(size_t(p) * ALPHABET + c) * WORDS + w] = a[w];
}
__global__ void all_anchor_distances(const U64 *masks, const int *alen, int m,
                                     const unsigned char *db, const int *len, int n,
                                     unsigned short *out) {
  int r = blockIdx.x * blockDim.x + threadIdx.x, a = blockIdx.y;
  if (r < n)
    out[size_t(a) * n + r] =
        distance(masks + size_t(a) * ALPHABET * WORDS, alen[a], db, n, r, len[r]);
}
__global__ void nearest_memberships(const unsigned short *dist, int n, int m, int *members) {
  using Sort = cub::BlockRadixSort<U32, THREADS, 4>;
  __shared__ typename Sort::TempStorage temp;
  int row = blockIdx.x;
  U32 keys[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int a = threadIdx.x * 4 + j;
    keys[j] = a < m ? (U32(dist[size_t(a) * n + row]) << 16) | U32(a) : ~U32(0);
  }
  Sort(temp).Sort(keys);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int rank = threadIdx.x * 4 + j;
    if (rank < MAXKB)
      members[size_t(row) * MAXKB + rank] = int(keys[j] & 65535);
  }
}
__global__ void posting_keys(const int *members, int n, int kb, U64 *keys, int *counts) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n * kb) {
    int row = i / kb, a = members[size_t(row) * MAXKB + i % kb];
    keys[i] = (U64(a) << 32) | U32(row);
    atomicAdd(counts + a, 1);
  }
}
__global__ void extract_postings(const U64 *keys, int count, int *ids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    ids[i] = int(U32(keys[i]));
}
__global__ void route_query(const U64 *masks, const int *qlen, const unsigned char *anchors,
                            const int *alen, int m, int kb, int *selected) {
  using Sort = cub::BlockRadixSort<U64, THREADS, 4>;
  __shared__ typename Sort::TempStorage temp;
  U64 keys[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int a = threadIdx.x * 4 + j;
    keys[j] =
        a < m ? (U64(distance(masks, *qlen, anchors, m, a, alen[a])) << 32) | U32(a) : INVALID;
  }
  Sort(temp).Sort(keys);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    int rank = threadIdx.x * 4 + j;
    if (rank < kb)
      selected[rank] = int(U32(keys[j]));
  }
}
__global__ void vote_postings(const int *selected, const int *offsets, const int *posting, int kb,
                              int *votes, int *stats) {
  int a = selected[blockIdx.y], start = offsets[a], end = offsets[a + 1];
  if (blockIdx.x == 0 && threadIdx.x == 0)
    atomicAdd(stats, end - start);
  for (int j = start + blockIdx.x * blockDim.x + threadIdx.x; j < end; j += gridDim.x * blockDim.x)
    atomicAdd(votes + posting[j], 1);
}
__global__ void vote_keys(const int *votes, const int *orig, int n, U64 *keys, int *stats) {
  using Reduce = cub::BlockReduce<int, THREADS>;
  __shared__ typename Reduce::TempStorage temp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int valid = i < n && votes[i] > 0;
  if (i < n)
    keys[i] = valid ? (U64(MAXKB - votes[i]) << 32) | U32(orig[i]) : INVALID;
  int count = Reduce(temp).Sum(valid);
  if (threadIdx.x == 0)
    atomicAdd(stats + 1, count);
}
__global__ void select_rows(const U64 *sorted, const int *id_to_row, int cap, U32 *rows,
                            int *stats) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cap)
    rows[i] = sorted[i] == INVALID ? ~U32(0) : U32(id_to_row[U32(sorted[i])]);
  if (i == 0)
    stats[2] = min(stats[1], cap);
}
__global__ void gather_candidates(const U32 *rows, const unsigned char *db, const int *len,
                                  const int *ids, int n, int cap, unsigned char *packed, int *plen,
                                  int *pids) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cap)
    return;
  U32 row = rows[i];
  int l = row == ~U32(0) ? 0 : len[row];
  plen[i] = l;
  pids[i] = row == ~U32(0) ? -1 : ids[row];
  for (int j = 0; j < MAXLEN; ++j)
    packed[size_t(j) * cap + i] = j < l ? db[size_t(j) * n + row] : 0;
}
// The same distance-plus-block-selection kernel is used by scan and verification.
__global__ void distance_top_blocks(const U64 *masks, const int *qlen, const unsigned char *text,
                                    const int *len, const int *ids, int n, U64 *out) {
  using Sort = cub::BlockRadixSort<U64, THREADS, 1>;
  __shared__ typename Sort::TempStorage temp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  U64 key[1];
  key[0] = i < n && ids[i] >= 0
               ? (U64(distance(masks, *qlen, text, n, i, len[i])) << 32) | U32(ids[i])
               : INVALID;
  Sort(temp).Sort(key);
  if (threadIdx.x < TOPK)
    out[size_t(blockIdx.x) * TOPK + threadIdx.x] = key[0];
}
__global__ void merge_top_blocks(const U64 *in, int n, U64 *out) {
  using Sort = cub::BlockRadixSort<U64, THREADS, 1>;
  __shared__ typename Sort::TempStorage temp;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  U64 key[1] = {i < n ? in[i] : INVALID};
  Sort(temp).Sort(key);
  if (threadIdx.x < TOPK)
    out[size_t(blockIdx.x) * TOPK + threadIdx.x] = key[0];
}
__global__ void pair_tests(const U64 *masks, const int *plen, const unsigned char *text,
                           const int *tlen, int n, int *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    result[i] = distance(masks + size_t(i) * ALPHABET * WORDS, plen[i], text, n, i, tlen[i]);
}

__global__ void sampled_pairs(const U64 *masks, const int *plen, const int *rows,
                              const unsigned char *db, const int *db_len, int stride, int count,
                              int *result) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    int row = rows[i];
    result[i] =
        distance(masks + size_t(i) * ALPHABET * WORDS, plen[i], db, stride, row, db_len[row]);
  }
}

static int cpu_distance(const std::string &a, const std::string &b) {
  std::vector<int> prev(b.size() + 1), cur(b.size() + 1);
  std::iota(prev.begin(), prev.end(), 0);
  for (size_t i = 1; i <= a.size(); ++i) {
    cur[0] = int(i);
    for (size_t j = 1; j <= b.size(); ++j)
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + int(a[i - 1] != b[j - 1])});
    prev.swap(cur);
  }
  return prev.back();
}
struct Corpus {
  int n;
  std::vector<std::string> input;
  Device<unsigned char> chars;
  Device<int> lengths, ids, id_to_row;
  Corpus(const std::vector<std::string> &strings)
      : n(int(strings.size())), input(strings), chars(size_t(MAXLEN) * n), lengths(n), ids(n),
        id_to_row(n) {
    if (!n)
      throw std::runtime_error("empty corpus");
    std::vector<int> order(n), lens(n), inverse(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return strings[a].size() < strings[b].size(); });
    std::vector<unsigned char> soa(size_t(MAXLEN) * n, 0);
    for (int row = 0; row < n; ++row) {
      int id = order[row];
      lens[row] = int(strings[id].size());
      inverse[id] = row;
      for (int j = 0; j < lens[row]; ++j)
        soa[size_t(j) * n + row] = static_cast<unsigned char>(strings[id][j]);
    }
    CUDA(cudaMemcpy(chars.p, soa.data(), soa.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(lengths.p, lens.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(ids.p, order.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(id_to_row.p, inverse.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  }
};
// Audit real database storage and random full-query pairs outside timing.
static void check_sampled_pairs(const Corpus &db, const std::vector<std::string> &queries,
                                int count) {
  std::mt19937 rng(38191);
  std::vector<int> inverse(db.n), rows(count), qlen(count), expected(count), actual(count);
  std::vector<unsigned char> qs(size_t(MAXLEN) * count, 0);
  CUDA(cudaMemcpy(inverse.data(), db.id_to_row.p, db.n * sizeof(int), cudaMemcpyDeviceToHost));
  for (int i = 0; i < count; ++i) {
    int id = int(rng() % db.n), qi = int(rng() % queries.size());
    rows[i] = inverse[id];
    qlen[i] = int(queries[qi].size());
    for (int j = 0; j < qlen[i]; ++j)
      qs[size_t(j) * count + i] = static_cast<unsigned char>(queries[qi][j]);
    expected[i] = cpu_distance(queries[qi], db.input[id]);
  }
  Device<int> drows(count), dlens(count), result(count);
  Device<unsigned char> dqs(qs.size());
  Device<U64> masks(size_t(count) * ALPHABET * WORDS);
  CUDA(cudaMemcpy(drows.p, rows.data(), count * sizeof(int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dlens.p, qlen.data(), count * sizeof(int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dqs.p, qs.data(), qs.size(), cudaMemcpyHostToDevice));
  make_masks<<<count, 256>>>(dqs.p, dlens.p, count, masks.p);
  LAUNCH();
  sampled_pairs<<<(count + 255) / 256, 256>>>(masks.p, dlens.p, drows.p, db.chars.p, db.lengths.p,
                                              db.n, count, result.p);
  LAUNCH();
  CUDA(cudaMemcpy(actual.data(), result.p, count * sizeof(int), cudaMemcpyDeviceToHost));
  if (actual != expected)
    throw std::runtime_error("sampled real database/query CPU DP mismatch");
  log("sample-check passed " + std::to_string(count) +
      " deterministic real database/query pairs outside timing");
}
struct AnchorSetup {
  int m;
  Device<unsigned char> chars;
  Device<int> lengths, members;
  Device<U64> masks;
  double build_ms = 0;
  AnchorSetup(const Corpus &db, int count, unsigned seed)
      : m(count), chars(size_t(MAXLEN) * m), lengths(m), members(size_t(db.n) * MAXKB),
        masks(size_t(m) * ALPHABET * WORDS) {
    auto begin = Clock::now();
    std::vector<int> chosen(db.n);
    std::iota(chosen.begin(), chosen.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(chosen.begin(), chosen.end(), rng);
    std::vector<unsigned char> soa(size_t(MAXLEN) * m, 0);
    std::vector<int> len(m);
    for (int a = 0; a < m; ++a) {
      const auto &s = db.input[chosen[a]];
      len[a] = int(s.size());
      for (int j = 0; j < len[a]; ++j)
        soa[size_t(j) * m + a] = static_cast<unsigned char>(s[j]);
    }
    CUDA(cudaMemcpy(chars.p, soa.data(), soa.size(), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(lengths.p, len.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    make_masks<<<m, ALPHABET>>>(chars.p, lengths.p, m, masks.p);
    LAUNCH();
    Device<unsigned short> dist(size_t(db.n) * m);
    all_anchor_distances<<<dim3((db.n + THREADS - 1) / THREADS, m), THREADS>>>(
        masks.p, lengths.p, m, db.chars.p, db.lengths.p, db.n, dist.p);
    LAUNCH();
    nearest_memberships<<<db.n, THREADS>>>(dist.p, db.n, m, members.p);
    LAUNCH();
    CUDA(cudaDeviceSynchronize());
    build_ms = ms(begin);
  }
};
struct Index {
  int kb;
  Device<int> offsets, postings;
  double build_ms = 0;
  size_t build_temporary_bytes = 0;
  Index(const Corpus &db, const AnchorSetup &anchors, int k)
      : kb(k), offsets(anchors.m + 1), postings(size_t(db.n) * k) {
    auto begin = Clock::now();
    int count = db.n * k;
    Device<U64> in(count), out(count);
    Device<int> counts(anchors.m + 1);
    CUDA(cudaMemset(counts.p, 0, (anchors.m + 1) * sizeof(int)));
    posting_keys<<<(count + 255) / 256, 256>>>(anchors.members.p, db.n, k, in.p, counts.p);
    LAUNCH();
    size_t sort_bytes = 0, sum_bytes = 0;
    CUDA(cub::DeviceRadixSort::SortKeys(nullptr, sort_bytes, in.p, out.p, count));
    CUDA(cub::DeviceScan::ExclusiveSum(nullptr, sum_bytes, counts.p, offsets.p, anchors.m + 1));
    Device<unsigned char> temp(std::max(sort_bytes, sum_bytes));
    build_temporary_bytes =
        size_t(count) * sizeof(U64) * 2 + size_t(anchors.m + 1) * sizeof(int) + temp.n;
    CUDA(cub::DeviceRadixSort::SortKeys(temp.p, sort_bytes, in.p, out.p, count));
    CUDA(cub::DeviceScan::ExclusiveSum(temp.p, sum_bytes, counts.p, offsets.p, anchors.m + 1));
    extract_postings<<<(count + 255) / 256, 256>>>(out.p, count, postings.p);
    LAUNCH();
    CUDA(cudaDeviceSynchronize());
    build_ms = ms(begin);
  }
};
struct Result {
  std::array<U64, TOPK> keys;
  double host_ms = 0, gpu_ms = 0, route_ms = 0, pool_ms = 0, gather_ms = 0, verify_ms = 0,
         output_ms = 0;
  int posting_count = 0, union_count = 0, candidate_count = 0;
};
struct Engine {
  const Corpus &db;
  const AnchorSetup &anchors;
  cudaStream_t stream;
  Events ev;
  Device<unsigned char> query, packed, temp;
  Device<int> qlen, selected, votes, stats, packed_len, packed_ids;
  Device<U64> masks, keys_a, keys_b, top_a, top_b;
  Device<U32> rows_a, rows_b;
  unsigned char *host_query = nullptr;
  int *host_len = nullptr;
  int *host_stats = nullptr;
  U64 *host_top = nullptr;
  int maxcap;
  Engine(const Corpus &d, const AnchorSetup &a, int cap)
      : db(d), anchors(a), query(MAXLEN), packed(size_t(MAXLEN) * cap), qlen(1), selected(MAXKB),
        votes(d.n), stats(3), packed_len(cap), packed_ids(cap), masks(ALPHABET * WORDS),
        keys_a(d.n), keys_b(d.n), top_a(size_t((d.n + 255) / 256) * TOPK),
        top_b(size_t((d.n + 255) / 256) * TOPK), rows_a(cap), rows_b(cap), maxcap(cap) {
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMallocHost(&host_query, MAXLEN));
    CUDA(cudaMallocHost(&host_len, sizeof(int)));
    CUDA(cudaMallocHost(&host_stats, 3 * sizeof(int)));
    CUDA(cudaMallocHost(&host_top, TOPK * sizeof(U64)));
    size_t b1 = 0, b2 = 0;
    CUDA(cub::DeviceRadixSort::SortKeys(nullptr, b1, keys_a.p, keys_b.p, db.n, 0, 64, stream));
    CUDA(cub::DeviceRadixSort::SortKeys(nullptr, b2, rows_a.p, rows_b.p, maxcap, 0, 32, stream));
    temp.resize(std::max(b1, b2));
  }
  size_t workspace_bytes() const {
    return query.n + packed.n + temp.n +
           (qlen.n + selected.n + votes.n + stats.n + packed_len.n + packed_ids.n) * sizeof(int) +
           (masks.n + keys_a.n + keys_b.n + top_a.n + top_b.n) * sizeof(U64) +
           (rows_a.n + rows_b.n) * sizeof(U32);
  }
  ~Engine() {
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFreeHost(host_query);
    cudaFreeHost(host_len);
    cudaFreeHost(host_stats);
    cudaFreeHost(host_top);
  }
  void prepare(const std::string &q) {
    *host_len = int(q.size());
    std::memset(host_query, 0, MAXLEN);
    std::memcpy(host_query, q.data(), q.size());
    CUDA(cudaMemcpyAsync(query.p, host_query, MAXLEN, cudaMemcpyHostToDevice, stream));
    CUDA(cudaMemcpyAsync(qlen.p, host_len, sizeof(int), cudaMemcpyHostToDevice, stream));
    make_masks<<<1, ALPHABET, 0, stream>>>(query.p, qlen.p, 1, masks.p);
    LAUNCH();
  }
  const U64 *top(const unsigned char *chars, const int *len, const int *ids, int n) {
    int blocks = (n + 255) / 256;
    distance_top_blocks<<<blocks, 256, 0, stream>>>(masks.p, qlen.p, chars, len, ids, n, top_a.p);
    LAUNCH();
    int count = blocks * TOPK;
    U64 *in = top_a.p;
    U64 *out = top_b.p;
    while (count > TOPK) {
      blocks = (count + 255) / 256;
      merge_top_blocks<<<blocks, 256, 0, stream>>>(in, count, out);
      LAUNCH();
      count = blocks * TOPK;
      std::swap(in, out);
    }
    return in;
  }
  double elapsed(int a, int b) {
    float f;
    CUDA(cudaEventElapsedTime(&f, ev.e[a], ev.e[b]));
    return f;
  }
  Result scan(const std::string &q) {
    Result r;
    auto start = Clock::now();
    CUDA(cudaEventRecord(ev.e[0], stream));
    prepare(q);
    const U64 *out = top(db.chars.p, db.lengths.p, db.ids.p, db.n);
    CUDA(cudaMemcpyAsync(host_top, out, TOPK * sizeof(U64), cudaMemcpyDeviceToHost, stream));
    CUDA(cudaEventRecord(ev.e[5], stream));
    CUDA(cudaEventSynchronize(ev.e[5]));
    r.host_ms = ms(start);
    r.gpu_ms = elapsed(0, 5);
    std::copy_n(host_top, TOPK, r.keys.begin());
    r.candidate_count = db.n;
    return r;
  }
  Result ann(const std::string &q, const Index &index, int cap) {
    Result r;
    auto start = Clock::now();
    CUDA(cudaEventRecord(ev.e[0], stream));
    prepare(q);
    route_query<<<1, THREADS, 0, stream>>>(masks.p, qlen.p, anchors.chars.p, anchors.lengths.p,
                                           anchors.m, index.kb, selected.p);
    LAUNCH();
    CUDA(cudaEventRecord(ev.e[1], stream));
    CUDA(cudaMemsetAsync(votes.p, 0, db.n * sizeof(int), stream));
    CUDA(cudaMemsetAsync(stats.p, 0, 3 * sizeof(int), stream));
    vote_postings<<<dim3(16, index.kb), 256, 0, stream>>>(
        selected.p, index.offsets.p, index.postings.p, index.kb, votes.p, stats.p);
    LAUNCH();
    vote_keys<<<(db.n + 255) / 256, 256, 0, stream>>>(votes.p, db.ids.p, db.n, keys_a.p, stats.p);
    LAUNCH();
    size_t bytes = temp.n;
    CUDA(cub::DeviceRadixSort::SortKeys(temp.p, bytes, keys_a.p, keys_b.p, db.n, 0, 64, stream));
    select_rows<<<(cap + 255) / 256, 256, 0, stream>>>(keys_b.p, db.id_to_row.p, cap, rows_a.p,
                                                       stats.p);
    LAUNCH();
    CUDA(cudaEventRecord(ev.e[2], stream));
    bytes = temp.n;
    CUDA(cub::DeviceRadixSort::SortKeys(temp.p, bytes, rows_a.p, rows_b.p, cap, 0, 32, stream));
    gather_candidates<<<(cap + 255) / 256, 256, 0, stream>>>(rows_b.p, db.chars.p, db.lengths.p,
                                                             db.ids.p, db.n, cap, packed.p,
                                                             packed_len.p, packed_ids.p);
    LAUNCH();
    CUDA(cudaEventRecord(ev.e[3], stream));
    const U64 *out = top(packed.p, packed_len.p, packed_ids.p, cap);
    CUDA(cudaEventRecord(ev.e[4], stream));
    CUDA(cudaMemcpyAsync(host_top, out, TOPK * sizeof(U64), cudaMemcpyDeviceToHost, stream));
    CUDA(cudaMemcpyAsync(host_stats, stats.p, 3 * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA(cudaEventRecord(ev.e[5], stream));
    CUDA(cudaEventSynchronize(ev.e[5]));
    r.host_ms = ms(start);
    r.gpu_ms = elapsed(0, 5);
    r.route_ms = elapsed(0, 1);
    r.pool_ms = elapsed(1, 2);
    r.gather_ms = elapsed(2, 3);
    r.verify_ms = elapsed(3, 4);
    r.output_ms = elapsed(4, 5);
    std::copy_n(host_top, TOPK, r.keys.begin());
    r.posting_count = host_stats[0];
    r.union_count = host_stats[1];
    r.candidate_count = host_stats[2];
    return r;
  }
};

static std::vector<U64> cpu_top(const std::vector<std::string> &db, const std::string &q) {
  std::vector<U64> keys(db.size());
  for (size_t i = 0; i < db.size(); ++i)
    keys[i] = (U64(cpu_distance(q, db[i])) << 32) | U32(i);
  std::partial_sort(keys.begin(), keys.begin() + std::min<size_t>(TOPK, keys.size()), keys.end());
  keys.resize(std::min<size_t>(TOPK, keys.size()));
  return keys;
}
static void check_scan(const Corpus &db, const std::string &q, const Result &r) {
  auto exact = cpu_top(db.input, q);
  for (size_t i = 0; i < exact.size(); ++i)
    if (r.keys[i] != exact[i])
      throw std::runtime_error("GPU scan/CPU DP top10 mismatch at rank " + std::to_string(i));
}
static std::pair<double, double> recalls(const Result &scan, const Result &ann) {
  int hits = 0, strict_hits = 0, strict_total = 0, boundary_hits = 0;
  U32 cutoff = U32(scan.keys[TOPK - 1] >> 32);
  for (U64 k : scan.keys)
    if (U32(k >> 32) < cutoff)
      ++strict_total;
  for (U64 a : ann.keys) {
    if (a == INVALID)
      continue;
    bool match = false;
    for (U64 b : scan.keys)
      if (U32(a) == U32(b)) {
        match = true;
        break;
      }
    if (match)
      ++hits;
    if (U32(a >> 32) < cutoff && match)
      ++strict_hits;
    if (U32(a >> 32) == cutoff)
      ++boundary_hits;
  }
  return {hits / double(TOPK),
          (strict_hits + std::min(boundary_hits, TOPK - strict_total)) / double(TOPK)};
}
static std::vector<std::string> read_lines(const std::string &path, int limit) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot read " + path);
  std::vector<std::string> out;
  std::string s;
  int line = 0;
  while (std::getline(f, s)) {
    ++line;
    if (!s.empty() && s.back() == '\r')
      s.pop_back();
    if (s.empty() || s.size() > MAXLEN)
      throw std::runtime_error("line length outside 1..256 at " + path + ":" +
                               std::to_string(line));
    for (unsigned char c : s)
      if (c < 32 || c > 126)
        throw std::runtime_error("file input must be printable ASCII");
    out.push_back(s);
    if (limit > 0 && int(out.size()) == limit)
      break;
  }
  if (out.empty())
    throw std::runtime_error("empty file " + path);
  if (limit > 0 && int(out.size()) != limit)
    throw std::runtime_error("file has fewer than requested records: " + path);
  return out;
}
struct Options {
  std::string db, queries, output, workload = "unspecified";
  int n = 100000, m = 512, repeats = 3, warmup = 3, query_limit = 0, sample_check = 0;
  unsigned seed = 42;
  std::vector<int> kb = {4, 8, 16};
  std::vector<double> fractions = {.02, .05};
  bool selftest = false, cpucheck = false;
};
static std::vector<std::string> split(const std::string &s) {
  std::vector<std::string> v;
  std::stringstream st(s);
  std::string x;
  while (std::getline(st, x, ','))
    v.push_back(x);
  return v;
}
static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto value = [&]() {
      if (i + 1 >= argc)
        throw std::runtime_error("missing value for " + a);
      return std::string(argv[++i]);
    };
    if (a == "--db")
      o.db = value();
    else if (a == "--queries")
      o.queries = value();
    else if (a == "--output")
      o.output = value();
    else if (a == "--workload")
      o.workload = value();
    else if (a == "--n")
      o.n = std::stoi(value());
    else if (a == "--m")
      o.m = std::stoi(value());
    else if (a == "--repeats")
      o.repeats = std::stoi(value());
    else if (a == "--warmup")
      o.warmup = std::stoi(value());
    else if (a == "--query-limit")
      o.query_limit = std::stoi(value());
    else if (a == "--sample-check")
      o.sample_check = std::stoi(value());
    else if (a == "--seed")
      o.seed = unsigned(std::stoul(value()));
    else if (a == "--kb") {
      o.kb.clear();
      for (auto &s : split(value()))
        o.kb.push_back(std::stoi(s));
    } else if (a == "--candidate-frac") {
      o.fractions.clear();
      for (auto &s : split(value()))
        o.fractions.push_back(std::stod(s));
    } else if (a == "--self-test")
      o.selftest = true;
    else if (a == "--cpu-check")
      o.cpucheck = true;
    else if (a == "--help") {
      std::cout << "Usage: soft_voronoi_bench --db FILE --queries FILE [--n 100000 --m 512 --kb "
                   "4,8,16 --candidate-frac .02,.05 --repeats 3 --warmup 3 --query-limit Q --seed "
                   "42 --output FILE --workload NAME --cpu-check --sample-check 64]\n       "
                   "soft_voronoi_bench --self-test\n";
      std::exit(0);
    } else
      throw std::runtime_error("unknown argument " + a);
  }
  if (o.selftest)
    return o;
  if (o.db.empty() || o.queries.empty())
    throw std::runtime_error("--db and --queries are required");
  if (o.n < TOPK || o.n > 10000000 || o.m < MAXKB || o.m > 1024 || o.m > o.n)
    throw std::runtime_error("require 10<=n<=10000000 and 16<=m<=min(n,1024)");
  if (o.repeats < 1 || o.warmup < 0 || o.query_limit < 0 || o.sample_check < 0 ||
      o.sample_check > 4096)
    throw std::runtime_error("invalid repetition/query count");
  if (o.kb.empty() || o.fractions.empty())
    throw std::runtime_error("empty sweep");
  std::vector<int> seen;
  for (int k : o.kb) {
    if (k < 1 || k > MAXKB || std::find(seen.begin(), seen.end(), k) != seen.end())
      throw std::runtime_error("kb must be unique in 1..16");
    seen.push_back(k);
  }
  std::vector<double> fs;
  for (double x : o.fractions) {
    if (!(x > 0 && x <= 1) || std::find(fs.begin(), fs.end(), x) != fs.end())
      throw std::runtime_error("fractions must be unique in (0,1]");
    fs.push_back(x);
  }
  if (o.cpucheck && o.n > 4096)
    throw std::runtime_error("--cpu-check limited to n<=4096; use self-test before full benchmark");
  return o;
}

static std::vector<int> chosen_anchors(int n, int m, unsigned seed) {
  std::vector<int> x(n);
  std::iota(x.begin(), x.end(), 0);
  std::mt19937 r(seed);
  std::shuffle(x.begin(), x.end(), r);
  x.resize(m);
  return x;
}
// Independent small-fixture CPU audit of every candidate-pipeline boundary.
static void check_pipeline(const Corpus &db, const AnchorSetup &a, const Index &index,
                           Engine &engine, const std::string &q, int cap, unsigned seed,
                           const Result &ann) {
  const int n = db.n, m = a.m;
  auto anchor = chosen_anchors(n, m, seed);
  std::vector<int> physical_ids(n), membership(size_t(n) * MAXKB);
  CUDA(cudaMemcpy(physical_ids.data(), db.ids.p, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(membership.data(), a.members.p, membership.size() * sizeof(int),
                  cudaMemcpyDeviceToHost));
  std::vector<int> row_votes(n, 0);
  std::vector<std::vector<int>> cpu_post(m);
  std::vector<std::pair<int, int>> route;
  for (int j = 0; j < m; ++j)
    route.emplace_back(cpu_distance(q, db.input[anchor[j]]), j);
  std::sort(route.begin(), route.end());
  std::vector<int> selected(index.kb);
  CUDA(cudaMemcpy(selected.data(), engine.selected.p, index.kb * sizeof(int),
                  cudaMemcpyDeviceToHost));
  for (int j = 0; j < index.kb; ++j)
    if (selected[j] != route[j].second)
      throw std::runtime_error("CPU/GPU routing mismatch");
  for (int row = 0; row < n; ++row) {
    int id = physical_ids[row];
    std::vector<std::pair<int, int>> ds;
    for (int j = 0; j < m; ++j)
      ds.emplace_back(cpu_distance(db.input[id], db.input[anchor[j]]), j);
    std::sort(ds.begin(), ds.end());
    for (int j = 0; j < MAXKB; ++j)
      if (membership[size_t(row) * MAXKB + j] != ds[j].second)
        throw std::runtime_error("CPU/GPU membership mismatch");
    for (int j = 0; j < index.kb; ++j) {
      int p = ds[j].second;
      cpu_post[p].push_back(row);
      if (std::find(selected.begin(), selected.end(), p) != selected.end())
        ++row_votes[row];
    }
  }
  std::vector<int> offsets(m + 1), post(n * index.kb), votes(n);
  CUDA(cudaMemcpy(offsets.data(), index.offsets.p, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(
      cudaMemcpy(post.data(), index.postings.p, post.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(votes.data(), engine.votes.p, n * sizeof(int), cudaMemcpyDeviceToHost));
  if (offsets[0] != 0 || offsets[m] != n * index.kb)
    throw std::runtime_error("bad CSR endpoints");
  for (int j = 0; j < m; ++j) {
    if (offsets[j + 1] - offsets[j] != int(cpu_post[j].size()))
      throw std::runtime_error("CPU/GPU CSR length mismatch");
    for (size_t p = 0; p < cpu_post[j].size(); ++p)
      if (post[offsets[j] + p] != cpu_post[j][p])
        throw std::runtime_error("CPU/GPU CSR content mismatch");
  }
  if (votes != row_votes)
    throw std::runtime_error("CPU/GPU vote mismatch");
  int union_count = 0, posting_count = 0;
  std::vector<std::pair<std::pair<int, int>, int>> candidates;
  for (int row = 0; row < n; ++row) {
    posting_count += row_votes[row];
    if (row_votes[row]) {
      ++union_count;
      candidates.push_back({{-row_votes[row], physical_ids[row]}, row});
    }
  }
  std::sort(candidates.begin(), candidates.end());
  if (int(candidates.size()) > cap)
    candidates.resize(cap);
  std::vector<U32> rows(cap, ~U32(0));
  std::vector<U64> exact;
  for (size_t j = 0; j < candidates.size(); ++j) {
    int row = candidates[j].second, id = physical_ids[row];
    rows[j] = U32(row);
    exact.push_back((U64(cpu_distance(q, db.input[id])) << 32) | U32(id));
  }
  std::sort(rows.begin(), rows.end());
  std::vector<U32> got_rows(cap);
  CUDA(cudaMemcpy(got_rows.data(), engine.rows_b.p, cap * sizeof(U32), cudaMemcpyDeviceToHost));
  if (got_rows != rows)
    throw std::runtime_error("CPU/GPU selected candidate mismatch");
  std::sort(exact.begin(), exact.end());
  exact.resize(TOPK, INVALID);
  for (int j = 0; j < TOPK; ++j)
    if (ann.keys[j] != exact[j])
      throw std::runtime_error("CPU/GPU ANN verification mismatch");
  if (ann.union_count != union_count || ann.posting_count != posting_count ||
      ann.candidate_count != int(candidates.size()))
    throw std::runtime_error("CPU/GPU candidate statistics mismatch");
}
static std::string random_string(std::mt19937 &r, int length, int alphabet) {
  std::string s(length, '\0');
  for (char &c : s)
    c = char(r() % alphabet);
  return s;
}
static void self_test() {
  log("self-test: independent CPU DP versus GPU Myers");
  std::mt19937 rng(20261029);
  std::vector<std::string> a, b;
  const int boundaries[] = {0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256};
  for (int x : boundaries)
    for (int y : boundaries) {
      a.emplace_back(x, 'a');
      b.emplace_back(y, 'b');
      a.emplace_back(x, 'a');
      b.emplace_back(y, 'a');
      a.push_back(random_string(rng, x, 256));
      b.push_back(random_string(rng, y, 256));
    }
  for (int i = 0; i < 4096; ++i) {
    int alphabet = i % 3 == 0 ? 4 : (i % 3 == 1 ? 26 : 256);
    int l = rng() % 257, t = rng() % 257;
    a.push_back(random_string(rng, l, alphabet));
    b.push_back(random_string(rng, t, alphabet));
    if (i % 4 == 0) {
      b.back() = a.back();
      if (!b.back().empty())
        b.back()[rng() % b.back().size()] = char(rng() % alphabet);
    }
  }
  int n = int(a.size());
  std::vector<unsigned char> pa(size_t(MAXLEN) * n, 0), tb(size_t(MAXLEN) * n, 0);
  std::vector<int> al(n), bl(n), expected(n), got(n);
  for (int i = 0; i < n; ++i) {
    al[i] = int(a[i].size());
    bl[i] = int(b[i].size());
    for (int j = 0; j < al[i]; ++j)
      pa[size_t(j) * n + i] = static_cast<unsigned char>(a[i][j]);
    for (int j = 0; j < bl[i]; ++j)
      tb[size_t(j) * n + i] = static_cast<unsigned char>(b[i][j]);
    expected[i] = cpu_distance(a[i], b[i]);
  }
  Device<unsigned char> da(pa.size()), db(tb.size());
  Device<int> dl(n), dt(n), dr(n);
  Device<U64> dm(size_t(n) * ALPHABET * WORDS);
  CUDA(cudaMemcpy(da.p, pa.data(), pa.size(), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db.p, tb.data(), tb.size(), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dl.p, al.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(dt.p, bl.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  make_masks<<<n, 256>>>(da.p, dl.p, n, dm.p);
  LAUNCH();
  for (int repeat = 0; repeat < 8; ++repeat) {
    pair_tests<<<(n + 255) / 256, 256>>>(dm.p, dl.p, db.p, dt.p, n, dr.p);
    LAUNCH();
    CUDA(cudaMemcpy(got.data(), dr.p, n * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
      if (got[i] != expected[i])
        throw std::runtime_error("Myers mismatch pair=" + std::to_string(i) +
                                 " lengths=" + std::to_string(al[i]) + "," + std::to_string(bl[i]) +
                                 " expected=" + std::to_string(expected[i]) +
                                 " got=" + std::to_string(got[i]));
  }
  log("self-test: " + std::to_string(n) + " boundary/random pairs passed eight repetitions");
  std::vector<std::string> strings;
  for (int i = 0; i < 257; ++i) {
    int l = (i % 7 == 0 ? boundaries[i % 16] : int(rng() % 130));
    strings.push_back(random_string(rng, l, 4));
  }
  strings[1] = strings[2] = strings[0];
  Corpus corpus(strings);
  AnchorSetup anchors(corpus, 32, 42);
  Engine engine(corpus, anchors, corpus.n);
  const std::vector<std::string> qs = {"",
                                       std::string(64, '\0'),
                                       std::string(65, '\1'),
                                       std::string(127, '\2'),
                                       std::string(128, '\3'),
                                       random_string(rng, 129, 4),
                                       random_string(rng, 255, 4),
                                       random_string(rng, 256, 256)};
  for (int kb : {4, 8, 16}) {
    Index index(corpus, anchors, kb);
    for (size_t qi = 0; qi < qs.size(); ++qi) {
      int cap = qi % 2 ? 37 : corpus.n;
      auto scan = engine.scan(qs[qi]);
      check_scan(corpus, qs[qi], scan);
      auto ann = engine.ann(qs[qi], index, cap);
      check_pipeline(corpus, anchors, index, engine, qs[qi], cap, 42, ann);
    }
  }
  // Every one-byte object is its own nearest anchor: this proves the <10 pool
  // path, rather than relying on a random fixture to encounter INVALID tails.
  std::vector<std::string> tiny_strings;
  for (int i = 0; i < 16; ++i)
    tiny_strings.emplace_back(1, char(i));
  Corpus tiny(tiny_strings);
  AnchorSetup tiny_anchors(tiny, 16, 42);
  Index tiny_index(tiny, tiny_anchors, 1);
  Engine tiny_engine(tiny, tiny_anchors, 16);
  auto tiny_scan = tiny_engine.scan(tiny_strings[0]);
  auto tiny_ann = tiny_engine.ann(tiny_strings[0], tiny_index, 16);
  check_scan(tiny, tiny_strings[0], tiny_scan);
  check_pipeline(tiny, tiny_anchors, tiny_index, tiny_engine, tiny_strings[0], 16, 42, tiny_ann);
  if (tiny_ann.candidate_count != 1 || tiny_ann.union_count != 1)
    throw std::runtime_error("deterministic singleton candidate fixture failed");
  for (int i = 1; i < TOPK; ++i)
    if (tiny_ann.keys[i] != INVALID)
      throw std::runtime_error("singleton pool did not preserve INVALID output tail");
  CUDA(cudaDeviceSynchronize());
  log("self-test: complete GPU memberships/CSR/routing/votes/candidates/gather/top10 agree with "
      "CPU");
  std::cout << "{\"type\":\"self_test\",\"status\":\"pass\",\"pair_cases\":" << n
            << ",\"pair_repetitions\":8,\"pipeline_cases\":25}" << std::endl;
}
static void write_keys(std::ostream &out, const char *name, const std::array<U64, TOPK> &keys,
                       bool ids) {
  out << ",\"" << name << "\":[";
  for (int i = 0; i < TOPK; ++i) {
    if (i)
      out << ',';
    if (keys[i] == INVALID)
      out << -1;
    else
      out << (ids ? U32(keys[i]) : U32(keys[i] >> 32));
  }
  out << ']';
}
int main(int argc, char **argv) {
  try {
    std::cout << std::unitbuf;
    Options o = parse(argc, argv);
    if (o.selftest) {
      self_test();
      return 0;
    }
    cudaDeviceProp prop{};
    CUDA(cudaGetDeviceProperties(&prop, 0));
    int runtime = 0, driver = 0;
    CUDA(cudaRuntimeGetVersion(&runtime));
    CUDA(cudaDriverGetVersion(&driver));
    std::ofstream file;
    if (!o.output.empty()) {
      file.open(o.output);
      if (!file)
        throw std::runtime_error("cannot open output file");
    }
    std::ostream &out = o.output.empty() ? std::cout : file;
    out << std::setprecision(9) << std::unitbuf;
    auto strings = read_lines(o.db, o.n), queries = read_lines(o.queries, o.query_limit);
    int maxcap = TOPK;
    for (double f : o.fractions)
      maxcap = std::max(maxcap, std::min(o.n, int(std::ceil(o.n * f))));
    out << "{\"type\":\"metadata\",\"workload\":" << quote(o.workload) << ",\"n\":" << o.n
        << ",\"m\":" << o.m << ",\"queries\":" << queries.size() << ",\"repeats\":" << o.repeats
        << ",\"warmup\":" << o.warmup << ",\"seed\":" << o.seed << ",\"gpu\":" << quote(prop.name)
        << ",\"sm_count\":" << prop.multiProcessorCount << ",\"compute_capability\":"
        << quote(std::to_string(prop.major) + "." + std::to_string(prop.minor))
        << ",\"cuda_runtime\":" << runtime << ",\"cuda_driver\":" << driver
        << ",\"metric\":\"exact_global_levenshtein_bytes_max256\",\"timing\":\"batch1_query_H2D_to_"
           "top10_D2H_synchronized\",\"graph\":false,\"db\":"
        << quote(o.db) << ",\"query_file\":" << quote(o.queries) << "}\n";
    log("loading/layout " + std::to_string(o.n) + " objects; building " + std::to_string(o.m) +
        " anchors");
    auto loadstart = Clock::now();
    Corpus db(strings);
    double layout_ms = ms(loadstart);
    if (o.sample_check) {
      check_sampled_pairs(db, queries, o.sample_check);
      out << "{\"type\":\"sample_check\",\"status\":\"pass\",\"pairs\":" << o.sample_check << "}\n";
    }
    AnchorSetup anchors(db, o.m, o.seed);
    Engine engine(db, anchors, maxcap);
    const size_t base_bytes = size_t(MAXLEN + 3 * sizeof(int)) * o.n +
                              size_t(o.n) * MAXKB * sizeof(int) +
                              size_t(o.m) * (MAXLEN + sizeof(int) + ALPHABET * WORDS * sizeof(U64));
    const size_t anchor_peak = base_bytes + size_t(o.n) * o.m * sizeof(unsigned short);
    out << "{\"type\":\"setup\",\"layout_ms\":" << layout_ms
        << ",\"anchor_setup_ms\":" << anchors.build_ms
        << ",\"database_bytes\":" << size_t(MAXLEN + 3 * sizeof(int)) * o.n
        << ",\"membership_bytes\":" << size_t(o.n) * MAXKB * sizeof(int)
        << ",\"query_workspace_bytes\":" << engine.workspace_bytes()
        << ",\"anchor_distance_temporary_bytes\":" << size_t(o.n) * o.m * sizeof(unsigned short)
        << ",\"anchor_storage_bytes\":"
        << size_t(o.m) * (MAXLEN + sizeof(int) + ALPHABET * WORDS * sizeof(U64))
        << ",\"anchor_build_peak_requested_bytes\":" << anchor_peak << "}\n";
    for (int kb : o.kb) {
      Index index(db, anchors, kb);
      out << "{\"type\":\"build\",\"kb\":" << kb << ",\"anchor_setup_ms\":" << anchors.build_ms
          << ",\"csr_build_ms\":" << index.build_ms
          << ",\"posting_bytes\":" << (size_t(o.n) * kb + o.m + 1) * sizeof(int)
          << ",\"build_temporary_bytes\":" << index.build_temporary_bytes
          << ",\"resident_requested_bytes\":"
          << base_bytes + engine.workspace_bytes() + (size_t(o.n) * kb + o.m + 1) * sizeof(int)
          << ",\"peak_build_requested_bytes\":"
          << std::max(anchor_peak, base_bytes + engine.workspace_bytes() +
                                       (size_t(o.n) * kb + o.m + 1) * sizeof(int) +
                                       index.build_temporary_bytes)
          << "}\n";
      for (double fraction : o.fractions) {
        int cap = std::min(o.n, std::max(TOPK, int(std::ceil(o.n * fraction))));
        log("benchmark kb=" + std::to_string(kb) + " cap=" + std::to_string(cap));
        for (int w = 0; w < o.warmup; ++w) {
          const auto &q = queries[w % queries.size()];
          engine.scan(q);
          engine.ann(q, index, cap);
        }
        for (size_t qi = 0; qi < queries.size(); ++qi) {
          for (int rep = 0; rep < o.repeats; ++rep) {
            bool first = ((qi + rep) % 2) == 0;
            Result scan, ann;
            if (first) {
              scan = engine.scan(queries[qi]);
              ann = engine.ann(queries[qi], index, cap);
            } else {
              ann = engine.ann(queries[qi], index, cap);
              scan = engine.scan(queries[qi]);
            }
            if (o.cpucheck && rep == 0) {
              check_scan(db, queries[qi], scan);
              if (first)
                check_pipeline(db, anchors, index, engine, queries[qi], cap, o.seed, ann);
              else {
                auto repeated = engine.ann(queries[qi], index, cap);
                check_pipeline(db, anchors, index, engine, queries[qi], cap, o.seed, repeated);
              }
            }
            auto recall = recalls(scan, ann);
            out << "{\"type\":\"query\",\"workload\":" << quote(o.workload) << ",\"n\":" << o.n
                << ",\"m\":" << o.m << ",\"kb\":" << kb << ",\"candidate_frac\":" << fraction
                << ",\"candidate_cap\":" << cap << ",\"query\":" << qi
                << ",\"query_length\":" << queries[qi].size() << ",\"repeat\":" << rep
                << ",\"order\":" << quote(first ? "scan_ann" : "ann_scan")
                << ",\"scan_gpu_ms\":" << scan.gpu_ms << ",\"ann_gpu_ms\":" << ann.gpu_ms
                << ",\"scan_host_ms\":" << scan.host_ms << ",\"ann_host_ms\":" << ann.host_ms
                << ",\"route_ms\":" << ann.route_ms << ",\"pool_ms\":" << ann.pool_ms
                << ",\"gather_ms\":" << ann.gather_ms << ",\"verify_ms\":" << ann.verify_ms
                << ",\"output_ms\":" << ann.output_ms << ",\"recall10\":" << recall.first
                << ",\"tie_recall10\":" << recall.second
                << ",\"candidate_count\":" << ann.candidate_count
                << ",\"union_count\":" << ann.union_count
                << ",\"posting_count\":" << ann.posting_count;
            write_keys(out, "scan_ids", scan.keys, true);
            write_keys(out, "scan_distances", scan.keys, false);
            write_keys(out, "ann_ids", ann.keys, true);
            write_keys(out, "ann_distances", ann.keys, false);
            out << "}\n";
          }
          if ((qi + 1) % 32 == 0)
            log("completed " + std::to_string(qi + 1) + "/" + std::to_string(queries.size()) +
                " queries");
        }
      }
    }
    log("benchmark complete");
    return 0;
  } catch (const std::exception &e) {
    log(std::string("ERROR: ") + e.what());
    return 1;
  }
}
