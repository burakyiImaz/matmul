#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include "matmul.h"

static double max_abs_diff(const Matrix& X, const Matrix& Y)
{
    double max_d = 0.0;
    for (int i = 0; i < X.rows; ++i) {
        for (int j = 0; j < X.cols; ++j) {
            max_d = std::max(max_d, std::abs(X(i, j) - Y(i, j)));
        }
    }
    return max_d;
}

static void naive_multiply(const Matrix& A, const Matrix& B, Matrix& C)
{
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
}

int main()
{
    {
        const int M = 31;
        const int K = 29;
        const int N = 19;
        Matrix A(M, K);
        Matrix B(K, N);
        Matrix C_fast(M, N);
        Matrix C_ref(M, N);

        std::mt19937 check_rng(7);
        std::uniform_real_distribution<double> check_dist(-1.0, 1.0);
        for (int i = 0; i < M; ++i) {
            for (int k = 0; k < K; ++k) {
                A(i, k) = check_dist(check_rng);
            }
        }
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                B(k, j) = check_dist(check_rng);
            }
        }

        FastMatmul::multiply(A, B, C_fast);
        naive_multiply(A, B, C_ref);
        const double err = max_abs_diff(C_fast, C_ref);
        std::cout << "correctness max_abs_err=" << err << std::endl;
    }

    const int N = 1024;
    Matrix A(N, N);
    Matrix B(N, N);
    Matrix C(N, N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A(i, j) = dist(rng);
            B(i, j) = dist(rng);
        }
    }

    FastMatmul::multiply(A, B, C);

    constexpr int runs = 5;
    double total_sec = 0.0;

    for (int r = 0; r < runs; ++r) {
        auto start = std::chrono::high_resolution_clock::now();
        FastMatmul::multiply(A, B, C);
        auto end = std::chrono::high_resolution_clock::now();
        total_sec += std::chrono::duration<double>(end - start).count();
    }

    const double avg_sec = total_sec / runs;
    const double gflops = (2.0 * N * N * N) / (avg_sec * 1e9);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N=" << N << " average_time=" << avg_sec << " sec, "
              << "throughput=" << gflops << " GFLOPS, "
              << "sample C(0,0)=" << C(0,0) << std::endl;

}