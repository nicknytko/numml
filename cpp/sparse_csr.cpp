#include "sparse_csr.hpp"

/** @file This file contains contains stub code for each function that will dispatch
    to either a CPU-based implementation or a CUDA-optimized version. */

FUNC_IMPL_DISPATCH(torch::Tensor,
                   spgemv_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                   torch::Tensor x) {
    if (is_cuda(A_data)) {
        return spgemv_forward_cuda(A_rows, A_cols, A_data, A_col_ind, A_rowptr, x);
    } else {
        return spgemv_forward_cpu(A_rows, A_cols, A_data, A_col_ind, A_rowptr, x);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spgemv_backward,
                   torch::Tensor grad_z, int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                   torch::Tensor x) {
    if (is_cuda(grad_z)) {
        return spgemv_backward_cuda(grad_z, A_rows, A_cols, A_data, A_col_ind, A_rowptr, x);
    } else {
        return spgemv_backward_cpu(grad_z, A_rows, A_cols, A_data, A_col_ind, A_rowptr, x);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spgemm_forward,
                   int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    if (is_cuda(A_data)) {
        return spgemm_forward_cuda(A_rows, A_cols, A_data, A_indices, A_indptr,
                                   B_rows, B_cols, B_data, B_indices, B_indptr);
    } else {
        return spgemm_forward_cpu(A_rows, A_cols, A_data, A_indices, A_indptr,
                                  B_rows, B_cols, B_data, B_indices, B_indptr);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spgemm_backward,
                   torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
                   int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    if (is_cuda(grad_C)) {
        return spgemm_backward_cuda(grad_C, C_indices, C_indptr,
                                    A_rows, A_cols, A_data, A_indices, A_indptr,
                                    B_rows, B_cols, B_data, B_indices, B_indptr);
    } else {
        return spgemm_backward_cpu(grad_C, C_indices, C_indptr,
                                   A_rows, A_cols, A_data, A_indices, A_indptr,
                                   B_rows, B_cols, B_data, B_indices, B_indptr);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   csr_transpose_forward,
                   int A_rows, int A_columns,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_transpose_forward_cuda(A_rows, A_columns,
                                          A_data, A_indices, A_indptr);
    } else {
        return csr_transpose_forward_cpu(A_rows, A_columns,
                                         A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_transpose_backward,
                   torch::Tensor grad_At, torch::Tensor At_to_A_idx) {
    if (is_cuda(grad_At)) {
        return csr_transpose_backward_cuda(grad_At, At_to_A_idx);
    } else {
        return csr_transpose_backward_cpu(grad_At, At_to_A_idx);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   splincomb_forward,
                   int rows, int cols,
                   torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {

    if (is_cuda(A_data)) {
        return splincomb_forward_cuda(rows, cols,
                                      alpha, A_data, A_indices, A_indptr,
                                      beta, B_data, B_indices, B_indptr);
    } else {
        return splincomb_forward_cpu(rows, cols,
                                     alpha, A_data, A_indices, A_indptr,
                                     beta, B_data, B_indices, B_indptr);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   splincomb_backward,
                   int rows, int cols,
                   torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr,
                   torch::Tensor grad_C_data, torch::Tensor C_indices, torch::Tensor C_indptr) {

    if (is_cuda(A_data)) {
        return splincomb_backward_cuda(rows, cols,
                                       alpha, A_data, A_indices, A_indptr,
                                       beta, B_data, B_indices, B_indptr,
                                       grad_C_data, C_indices, C_indptr);
    } else {
        return splincomb_backward_cpu(rows, cols,
                                      alpha, A_data, A_indices, A_indptr,
                                      beta, B_data, B_indices, B_indptr,
                                      grad_C_data, C_indices, C_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   spdmm_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   torch::Tensor B) {

    TORCH_CHECK((A_data.device() == A_indices.device() &&
                 A_indices.device() == A_indptr.device() &&
                 A_indptr.device() == B.device()), "expected A and B to be on same device, but got A on (",
                A_data.device(), ", ", A_indices.device(), ", ", A_indptr.device(), ") and B on ", B.device());

    if (is_cuda(A_data)) {
        return spdmm_forward_cuda(A_rows, A_cols, A_data, A_indices, A_indptr, B);
    } else {
        return spdmm_forward_cpu(A_rows, A_cols, A_data, A_indices, A_indptr, B);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spdmm_backward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   torch::Tensor B, torch::Tensor grad_C) {

    TORCH_CHECK((A_data.device() == A_indices.device() &&
                 A_indices.device() == A_indptr.device() &&
                 A_indptr.device() == B.device()), "expected A and B to be on same device, but got A on (",
                A_data.device(), ", ", A_indices.device(), ", ", A_indptr.device(), ") and B on ", B.device());
    TORCH_CHECK(B.device() == grad_C.device(), "expected B and grad_C to be on same device, but got B on ",
                B.device(), " and grad_CC on ", grad_C.device());

    if (is_cuda(A_data)) {
        return spdmm_backward_cuda(A_rows, A_cols,
                                   A_data, A_indices, A_indptr, B, grad_C);
    } else {
        return spdmm_backward_cpu(A_rows, A_cols,
                                  A_data, A_indices, A_indptr, B, grad_C);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   sptrsv_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   bool lower, bool unit, torch::Tensor b) {
    if (is_cuda(A_data)) {
        return sptrsv_forward_cuda(A_rows, A_cols, A_data, A_indices, A_indptr,
                                   lower, unit, b);
    } else {
        return sptrsv_forward_cpu(A_rows, A_cols, A_data, A_indices, A_indptr,
                                  lower, unit, b);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   sptrsv_backward,
                   torch::Tensor grad_x, torch::Tensor x,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                   bool lower, bool unit, torch::Tensor b) {
    if (is_cuda(A_data)) {
        return sptrsv_backward_cuda(grad_x, x, A_rows, A_cols,
                                    A_data, A_indices, A_indptr, lower, unit, b);
    } else {
        return sptrsv_backward_cpu(grad_x, x, A_rows, A_cols,
                                   A_data, A_indices, A_indptr, lower, unit, b);

    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   splu,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    if (is_cuda(A_data)) {
        return splu_cuda(A_rows, A_cols, A_data, A_indices, A_indptr);
    } else {
        return splu_cpu(A_rows, A_cols, A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spsolve_backward,
                   torch::Tensor grad_x, torch::Tensor x,
                   int A_rows, int A_cols,
                   torch::Tensor Mt_data, torch::Tensor Mt_indices, torch::Tensor Mt_indptr,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    if (is_cuda(A_data)) {
        return spsolve_backward_cuda(grad_x, x, A_rows, A_cols,
                                     Mt_data, Mt_indices, Mt_indptr,
                                     A_data, A_indices, A_indptr);
    } else {
        return spsolve_backward_cpu(grad_x, x, A_rows, A_cols,
                                    Mt_data, Mt_indices, Mt_indptr,
                                    A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_to_dense_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_to_dense_forward_cuda(A_rows, A_cols,
                                         A_data, A_indices, A_indptr);
    } else {
        return csr_to_dense_forward_cpu(A_rows, A_cols,
                                        A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_to_dense_backward,
                   torch::Tensor grad_Ad,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_to_dense_backward_cuda(grad_Ad, A_rows, A_cols,
                                          A_data, A_indices, A_indptr);
    } else {
        return csr_to_dense_backward_cpu(grad_Ad, A_rows, A_cols,
                                         A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_row_sum_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_row_sum_forward_cuda(A_rows, A_cols,
                                        A_data, A_indices, A_indptr);
    } else {
        return csr_row_sum_forward_cpu(A_rows, A_cols,
                                       A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_row_sum_backward,
                   torch::Tensor grad_x,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_row_sum_backward_cuda(grad_x, A_rows, A_cols,
                                         A_data, A_indices, A_indptr);
    } else {
        return csr_row_sum_backward_cpu(grad_x, A_rows, A_cols,
                                        A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_extract_diagonal_forward,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_extract_diagonal_forward_cuda(A_rows, A_cols,
                                        A_data, A_indices, A_indptr);
    } else {
        return csr_extract_diagonal_forward_cpu(A_rows, A_cols,
                                       A_data, A_indices, A_indptr);
    }
}

FUNC_IMPL_DISPATCH(torch::Tensor,
                   csr_extract_diagonal_backward,
                   torch::Tensor grad_x,
                   int A_rows, int A_cols,
                   torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    if (is_cuda(A_data)) {
        return csr_extract_diagonal_backward_cuda(grad_x, A_rows, A_cols,
                                         A_data, A_indices, A_indptr);
    } else {
        return csr_extract_diagonal_backward_cpu(grad_x, A_rows, A_cols,
                                        A_data, A_indices, A_indptr);
    }
}

#define DEF_PYBIND(NAME, DESC)                                          \
    m.def(#NAME "_forward", & NAME##_forward, DESC " forward");         \
    m.def(#NAME "_backward", & NAME##_backward, DESC " backward");

#define DEF_PYBIND2(NAME, DESC)                 \
    m.def(#NAME, &NAME, DESC);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    DEF_PYBIND(spgemv, "SPGEMV");
    DEF_PYBIND(spgemm, "SPGEMM");
    DEF_PYBIND(csr_transpose, "CSR Transpose");
    DEF_PYBIND(splincomb, "Sparse linear combination");
    DEF_PYBIND(spdmm, "Sparse times dense matrix");
    DEF_PYBIND(sptrsv, "Sparse triangular solve");
    DEF_PYBIND2(splu, "Sparse LU decomposition");
    DEF_PYBIND2(spsolve_backward, "Sparse LU solve backward");
    DEF_PYBIND(csr_to_dense, "CSR to dense");
    DEF_PYBIND(csr_row_sum, "CSR row sum");
    DEF_PYBIND(csr_extract_diagonal, "CSR extract diagonal");
}
