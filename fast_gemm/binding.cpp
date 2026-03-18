#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

py:: array_t<double> matmul_py(
    py::array_t<double> A_np,
    py::array_t<double> B_np
)

{
    auto bufA = A_np.request();
    auto bufB = B_np.request();

    int M = bufA.shape[0];
    int K = bufA.shape[1];
    int N = bufB.shape[1];

    Matrix A(M, K);
    Matrix B(K, N); 
    Matrix C(M, N);


    double* ptrA = (double*) bufA.ptr;
    double* ptrB = (double*) bufB.ptr;

    // Copy data from numpy arrays to Matrix objects
    for(int i=0;i<M*K;i++) A.data[i] = ptrA[i];
    for(int i=0;i<K*N;i++) B.data[i] = ptrB[i];

    FastMatmul::multiply(A, B, C);

    // Create a numpy array for the result
    auto result = py::array_t<double>({M, N});
    auto bufC = result.request();
    double* ptrC = (double*) bufC.ptr;

    // Copy data from Matrix object to numpy array
    for(int i=0;i<M*N;i++) ptrC[i] = C.data[i];

    return result;

}