#include "matmul.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__APPLE__)
#include <vecLib/cblas.h>
#endif

namespace {

int parse_blas_min_dim_from_env()
{
    const char* env = std::getenv("FASTGEMM_BLAS_MIN_DIM");
    if (env == nullptr) {
        return 192;
    }
    char* end = nullptr;
    long v = std::strtol(env, &end, 10);
    if (end == env || *end != '\0') {
        return 192;
    }
    if (v < 1) {
        return 1;
    }
    if (v > 1'000'000) {
        return 1'000'000;
    }
    return static_cast<int>(v);
}

long long parse_blas_min_mnk_from_env()
{
    const char* env = std::getenv("FASTGEMM_BLAS_MIN_MNK");
    if (env == nullptr) {
        return 8LL * 1024LL * 1024LL;
    }
    char* end = nullptr;
    long long v = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0') {
        return 8LL * 1024LL * 1024LL;
    }
    if (v < 1) {
        return 1;
    }
    if (v > 1'000'000'000'000LL) {
        return 1'000'000'000'000LL;
    }
    return v;
}

int& blas_min_dim_storage()
{
    static int min_dim = parse_blas_min_dim_from_env();
    return min_dim;
}

long long& blas_min_mnk_storage()
{
    static long long min_mnk = parse_blas_min_mnk_from_env();
    return min_mnk;
}

bool should_use_blas(int M, int K, int N)
{
    const int min_dim = blas_min_dim_storage();
    const long long mnk = static_cast<long long>(M) * K * N;
    return (M >= min_dim && K >= min_dim && N >= min_dim) || (mnk >= blas_min_mnk_storage());
}

bool should_parallel_custom(int M, int K, int N)
{
    // Aggr more aggressive parallelization: even smaller problems benefit from threading.
    const long long mnk = static_cast<long long>(M) * K * N;
    return mnk >= 16LL * 1024LL * 1024LL;  // Lowered threshold for 512+ range
}

inline void scalar_cell(
    const double* A,
    const double* B,
    double* C,
    int i,
    int j,
    int K,
    int lda,
    int ldb,
    int ldc
)
{
    double sum = 0.0;
    const double* a = &A[i * lda];
    const double* b = &B[j];
    for (int k = 0; k < K; ++k) {
        sum += a[k] * (*b);
        b += ldb;
    }
    C[i * ldc + j] = sum;
}

} // namespace

void FastMatmul::multiply_raw(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N,
    int lda,
    int ldb,
    int ldc
)
{
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::invalid_argument("FastMatmul::multiply_raw: A/B/C must be non-null");
    }
    if (M < 0 || K < 0 || N < 0 || lda < 0 || ldb < 0 || ldc < 0) {
        throw std::invalid_argument("FastMatmul::multiply_raw: dimensions/strides must be non-negative");
    }
    if (M == 0 || K == 0 || N == 0) {
        return;
    }
    if (lda < K || ldb < N || ldc < N) {
        throw std::invalid_argument("FastMatmul::multiply_raw: invalid strides for row-major contiguous inputs");
    }

#if defined(__APPLE__)
    // For larger shapes, delegate to tuned system BLAS (multi-threaded on Apple).
    if (should_use_blas(M, K, N)) {
        cblas_dgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            M,
            N,
            K,
            1.0,
            A,
            lda,
            B,
            ldb,
            0.0,
            C,
            ldc
        );
        return;
    }
#endif

    const int MR = MicroKernel::MR;
    const int NR = MicroKernel::NR;
    const int i_full = (M / MR) * MR;
    const int j_full = (N / NR) * NR;
    const int j_tiles = j_full / NR;

    const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
    const bool parallel_custom = should_parallel_custom(M, K, N);
    const int num_threads = parallel_custom
        ? std::max(1, std::min<int>(static_cast<int>(hw), std::max(1, j_tiles)))
        : 1;

    auto tile_worker = [&](int tile_begin, int tile_end) {
        std::vector<double> packedB(static_cast<size_t>(K) * NR);
        for (int jt = tile_begin; jt < tile_end; ++jt) {
            const int j = jt * NR;
            // Fast B packing with memcpy per k
            for (int k = 0; k < K; ++k) {
                std::memcpy(&packedB[k * NR], &B[k * ldb + j], NR * sizeof(double));
            }
            // Compute tile
            for (int i = 0; i < i_full; i += MR) {
                MicroKernel::kernel(
                    K,
                    &A[i * lda],
                    packedB.data(),
                    &C[i * ldc + j],
                    ldc,
                    false
                );
            }
        }
    };

    if (j_tiles > 0) {
        if (num_threads == 1) {
            tile_worker(0, j_tiles);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(num_threads);
            for (int t = 0; t < num_threads; ++t) {
                const int begin = (j_tiles * t) / num_threads;
                const int end = (j_tiles * (t + 1)) / num_threads;
                if (begin < end) {
                    workers.emplace_back(tile_worker, begin, end);
                }
            }
            for (auto& th : workers) {
                th.join();
            }
        }
    }

    for (int i = 0; i < i_full; ++i) {
        for (int j = j_full; j < N; ++j) {
            scalar_cell(A, B, C, i, j, K, lda, ldb, ldc);
        }
    }

    for (int i = i_full; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            scalar_cell(A, B, C, i, j, K, lda, ldb, ldc);
        }
    }
}

void FastMatmul::multiply(const Matrix& A, const Matrix& B, Matrix& C)
{
    const int M = A.rows;
    const int K = A.cols;
    const int N = B.cols;

    if (K != B.rows) {
        throw std::invalid_argument("FastMatmul::multiply: A.cols must equal B.rows");
    }
    if (C.rows != M || C.cols != N) {
        throw std::invalid_argument("FastMatmul::multiply: C shape must be (A.rows, B.cols)");
    }

    multiply_raw(
        A.data.data(),
        B.data.data(),
        C.data.data(),
        M,
        K,
        N,
        A.cols,
        B.cols,
        C.cols
    );
}

void FastMatmul::set_blas_min_dim(int min_dim)
{
    if (min_dim < 1) {
        throw std::invalid_argument("FastMatmul::set_blas_min_dim: value must be >= 1");
    }
    blas_min_dim_storage() = min_dim;
}

int FastMatmul::get_blas_min_dim()
{
    return blas_min_dim_storage();
}

void FastMatmul::set_blas_min_mnk(long long min_mnk)
{
    if (min_mnk < 1) {
        throw std::invalid_argument("FastMatmul::set_blas_min_mnk: value must be >= 1");
    }
    blas_min_mnk_storage() = min_mnk;
}

long long FastMatmul::get_blas_min_mnk()
{
    return blas_min_mnk_storage();
}