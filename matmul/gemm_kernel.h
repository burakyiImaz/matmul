#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define FASTGEMM_RESTRICT __restrict__
#else
#define FASTGEMM_RESTRICT
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__)
#include <arm_neon.h>
#endif

class MicroKernel
{
public:

static constexpr int MR = 4;
#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
static constexpr int NR = 8;
#else
static constexpr int NR = 4;
#endif

static void kernel(
    int K,
    const double* FASTGEMM_RESTRICT A,
    const double* FASTGEMM_RESTRICT B,
    double* FASTGEMM_RESTRICT C,
    int ldc,
    bool accumulate = false
)
{
#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
    kernel_4x8(K, A, B, C, ldc, accumulate);
#elif defined(__aarch64__)
    kernel_4x4_neon(K, A, B, C, ldc, accumulate);
#else
    kernel_4x4_scalar(K, A, B, C, ldc, accumulate);
#endif
}

private:

static void kernel_4x8(
    int K,
    const double* FASTGEMM_RESTRICT A,
    const double* FASTGEMM_RESTRICT B,
    double* FASTGEMM_RESTRICT C,
    int ldc,
    bool accumulate = false
)
{

#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)

    __m256d c00 = accumulate ? _mm256_loadu_pd(&C[0*ldc]) : _mm256_setzero_pd();
    __m256d c01 = accumulate ? _mm256_loadu_pd(&C[0*ldc+4]) : _mm256_setzero_pd();

    __m256d c10 = accumulate ? _mm256_loadu_pd(&C[1*ldc]) : _mm256_setzero_pd();
    __m256d c11 = accumulate ? _mm256_loadu_pd(&C[1*ldc+4]) : _mm256_setzero_pd();

    __m256d c20 = accumulate ? _mm256_loadu_pd(&C[2*ldc]) : _mm256_setzero_pd();
    __m256d c21 = accumulate ? _mm256_loadu_pd(&C[2*ldc+4]) : _mm256_setzero_pd();

    __m256d c30 = accumulate ? _mm256_loadu_pd(&C[3*ldc]) : _mm256_setzero_pd();
    __m256d c31 = accumulate ? _mm256_loadu_pd(&C[3*ldc+4]) : _mm256_setzero_pd();


    const double* a0_ptr = A;
    const double* a1_ptr = A + K;
    const double* a2_ptr = A + 2 * K;
    const double* a3_ptr = A + 3 * K;
    const double* b_ptr = B;

    for (int k = 0; k < K; ++k)
    {
        __m256d b0 = _mm256_loadu_pd(b_ptr);
        __m256d b1 = _mm256_loadu_pd(b_ptr + 4);

        __m256d a0 = _mm256_broadcast_sd(a0_ptr++);
        __m256d a1 = _mm256_broadcast_sd(a1_ptr++);
        __m256d a2 = _mm256_broadcast_sd(a2_ptr++);
        __m256d a3 = _mm256_broadcast_sd(a3_ptr++);


        c00 = _mm256_fmadd_pd(a0,b0,c00);
        c01 = _mm256_fmadd_pd(a0,b1,c01);

        c10 = _mm256_fmadd_pd(a1,b0,c10);
        c11 = _mm256_fmadd_pd(a1,b1,c11);

        c20 = _mm256_fmadd_pd(a2,b0,c20);
        c21 = _mm256_fmadd_pd(a2,b1,c21);

        c30 = _mm256_fmadd_pd(a3,b0,c30);
        c31 = _mm256_fmadd_pd(a3,b1,c31);

        b_ptr += 8;

    }


    _mm256_storeu_pd(&C[0*ldc],c00);
    _mm256_storeu_pd(&C[0*ldc+4],c01);

    _mm256_storeu_pd(&C[1*ldc],c10);
    _mm256_storeu_pd(&C[1*ldc+4],c11);

    _mm256_storeu_pd(&C[2*ldc],c20);
    _mm256_storeu_pd(&C[2*ldc+4],c21);

    _mm256_storeu_pd(&C[3*ldc],c30);
    _mm256_storeu_pd(&C[3*ldc+4],c31);

#else

    (void)K;
    (void)A;
    (void)B;
    (void)C;
    (void)ldc;

#endif

}

static void kernel_4x4_neon(
    int K,
    const double* FASTGEMM_RESTRICT A,
    const double* FASTGEMM_RESTRICT B,
    double* FASTGEMM_RESTRICT C,
    int ldc,
    bool accumulate = false
)
{
#if defined(__aarch64__)
    float64x2_t c00 = accumulate ? vld1q_f64(&C[0*ldc]) : vdupq_n_f64(0.0);
    float64x2_t c01 = accumulate ? vld1q_f64(&C[0*ldc+2]) : vdupq_n_f64(0.0);
    float64x2_t c10 = accumulate ? vld1q_f64(&C[1*ldc]) : vdupq_n_f64(0.0);
    float64x2_t c11 = accumulate ? vld1q_f64(&C[1*ldc+2]) : vdupq_n_f64(0.0);
    float64x2_t c20 = accumulate ? vld1q_f64(&C[2*ldc]) : vdupq_n_f64(0.0);
    float64x2_t c21 = accumulate ? vld1q_f64(&C[2*ldc+2]) : vdupq_n_f64(0.0);
    float64x2_t c30 = accumulate ? vld1q_f64(&C[3*ldc]) : vdupq_n_f64(0.0);
    float64x2_t c31 = accumulate ? vld1q_f64(&C[3*ldc+2]) : vdupq_n_f64(0.0);

    const double* a0_ptr = A;
    const double* a1_ptr = A + K;
    const double* a2_ptr = A + 2 * K;
    const double* a3_ptr = A + 3 * K;
    const double* b_ptr = B;

    for (int k = 0; k < K; ++k) {
        const float64x2_t b0 = vld1q_f64(b_ptr);
        const float64x2_t b1 = vld1q_f64(b_ptr + 2);

        const float64x2_t a0 = vdupq_n_f64(*a0_ptr++);
        const float64x2_t a1 = vdupq_n_f64(*a1_ptr++);
        const float64x2_t a2 = vdupq_n_f64(*a2_ptr++);
        const float64x2_t a3 = vdupq_n_f64(*a3_ptr++);

        c00 = vmlaq_f64(c00, b0, a0);
        c01 = vmlaq_f64(c01, b1, a0);
        c10 = vmlaq_f64(c10, b0, a1);
        c11 = vmlaq_f64(c11, b1, a1);
        c20 = vmlaq_f64(c20, b0, a2);
        c21 = vmlaq_f64(c21, b1, a2);
        c30 = vmlaq_f64(c30, b0, a3);
        c31 = vmlaq_f64(c31, b1, a3);

        b_ptr += 4;
    }

    vst1q_f64(&C[0 * ldc], c00);
    vst1q_f64(&C[0 * ldc + 2], c01);
    vst1q_f64(&C[1 * ldc], c10);
    vst1q_f64(&C[1 * ldc + 2], c11);
    vst1q_f64(&C[2 * ldc], c20);
    vst1q_f64(&C[2 * ldc + 2], c21);
    vst1q_f64(&C[3 * ldc], c30);
    vst1q_f64(&C[3 * ldc + 2], c31);
#else
    (void)K;
    (void)A;
    (void)B;
    (void)C;
    (void)ldc;
    (void)accumulate;
#endif
}

static void kernel_4x4_scalar(
    int K,
    const double* FASTGEMM_RESTRICT A,
    const double* FASTGEMM_RESTRICT B,
    double* FASTGEMM_RESTRICT C,
    int ldc,
    bool accumulate = false
)
{
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            double sum = accumulate ? C[r * ldc + c] : 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[r * K + k] * B[k * 4 + c];
            }
            C[r * ldc + c] = sum;
        }
    }
}

};