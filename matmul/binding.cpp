#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "matmul.h"

namespace py = pybind11;

py::array_t<double> matmul_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> B_np
)
{
    auto bufA = A_np.request();
    auto bufB = B_np.request();

    if (bufA.ndim != 2 || bufB.ndim != 2) {
        throw std::invalid_argument("matmul expects two 2D arrays");
    }

    int M = bufA.shape[0];
    int K = bufA.shape[1];
    int BK = bufB.shape[0];
    int N = bufB.shape[1];

    if (K != BK) {
        throw std::invalid_argument("matmul shape mismatch: A.shape[1] must equal B.shape[0]");
    }

    const double* ptrA = static_cast<const double*>(bufA.ptr);
    const double* ptrB = static_cast<const double*>(bufB.ptr);

    auto result = py::array_t<double>({M,N});
    auto bufC = result.request();
    double* ptrC = static_cast<double*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        FastMatmul::multiply_raw(
            ptrA,
            ptrB,
            ptrC,
            M,
            K,
            N,
            K,
            N,
            N
        );
    }

    return result;
}

void matmul_out_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> A_np,
    py::array_t<double, py::array::c_style | py::array::forcecast> B_np,
    py::array_t<double, py::array::c_style> C_np
)
{
    auto bufA = A_np.request();
    auto bufB = B_np.request();
    auto bufC = C_np.request();

    if (bufA.ndim != 2 || bufB.ndim != 2 || bufC.ndim != 2) {
        throw std::invalid_argument("matmul_out expects three 2D arrays");
    }

    const int M = static_cast<int>(bufA.shape[0]);
    const int K = static_cast<int>(bufA.shape[1]);
    const int BK = static_cast<int>(bufB.shape[0]);
    const int N = static_cast<int>(bufB.shape[1]);

    if (K != BK) {
        throw std::invalid_argument("matmul_out shape mismatch: A.shape[1] must equal B.shape[0]");
    }
    if (bufC.shape[0] != M || bufC.shape[1] != N) {
        throw std::invalid_argument("matmul_out output shape mismatch: C must be (A.shape[0], B.shape[1])");
    }

    const double* ptrA = static_cast<const double*>(bufA.ptr);
    const double* ptrB = static_cast<const double*>(bufB.ptr);
    double* ptrC = static_cast<double*>(bufC.ptr);

    {
        py::gil_scoped_release release;
        FastMatmul::multiply_raw(
            ptrA,
            ptrB,
            ptrC,
            M,
            K,
            N,
            K,
            N,
            N
        );
    }
}

PYBIND11_MODULE(fastgemm, m)
{
    m.def("matmul", &matmul_py);
    m.def("matmul_out", &matmul_out_py);
    m.def("set_blas_min_dim", &FastMatmul::set_blas_min_dim);
    m.def("get_blas_min_dim", &FastMatmul::get_blas_min_dim);
    m.def("set_blas_min_mnk", &FastMatmul::set_blas_min_mnk);
    m.def("get_blas_min_mnk", &FastMatmul::get_blas_min_mnk);
}