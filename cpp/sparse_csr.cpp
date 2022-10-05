#include "sparse_csr.hpp"

/** @file This file contains contains stub code for each function that will dispatch
    to either a CPU-based implementation or a CUDA-optimized version. */

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spgemv_forward,
                   int A_rows, int A_cols, torch::Tensor alpha,
                   torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                   torch::Tensor x, torch::Tensor beta, torch::Tensor y) {
    if (is_cuda(A_data)) {
        return spgemv_forward_cuda(A_rows, A_cols, alpha, A_data, A_col_ind, A_rowptr, x, beta, y);
    } else {
        return spgemv_forward_cpu(A_rows, A_cols, alpha, A_data, A_col_ind, A_rowptr, x, beta, y);
    }
}

FUNC_IMPL_DISPATCH(std::vector<torch::Tensor>,
                   spgemv_backward,
                   torch::Tensor grad_z, int A_rows, int A_cols, torch::Tensor alpha,
                   torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                   torch::Tensor x, torch::Tensor beta, torch::Tensor y) {
    if (is_cuda(grad_z)) {
        return spgemv_backward_cuda(grad_z, A_rows, A_cols, alpha, A_data, A_col_ind, A_rowptr,
                                    x, beta, y);
    } else {
        return spgemv_backward_cpu(grad_z, A_rows, A_cols, alpha, A_data, A_col_ind, A_rowptr,
                                   x, beta, y);
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spgemv_forward", &spgemv_forward, "SPGEMV forward");
    m.def("spgemv_backward", &spgemv_backward, "SPGEMV backward");

    m.def("spgemm_forward", &spgemm_forward, "SPGEMM forward");
    m.def("spgemm_backward", &spgemm_backward, "SPGEMM backward");

    m.def("csr_transpose_forward", &csr_transpose_forward, "CSR Transpose forward");
    m.def("csr_transpose_backward", &csr_transpose_backward, "CSR Transpose backward");
}