#pragma once
#include "matrix.h"
#include "gemm_kernel.h"

class FastMatmul
{

public:

static void multiply(const Matrix& A, const Matrix& B, Matrix& C);

static void multiply_raw(
	const double* A,
	const double* B,
	double* C,
	int M,
	int K,
	int N,
	int lda,
	int ldb,
	int ldc
);

static void set_blas_min_dim(int min_dim);
static int get_blas_min_dim();
static void set_blas_min_mnk(long long min_mnk);
static long long get_blas_min_mnk();

};
