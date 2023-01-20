#ifndef SPARSE_CSR_HPP_
#define SPARSE_CSR_HPP_

#include <torch/extension.h>
#include <vector>
#include <stdexcept>

/* Define dispatch, cpu, and cuda versions of all functions */
#define FUNC_DEF(ret, name, ...) \
    ret name##_cpu(__VA_ARGS__); \
    ret name##_cuda(__VA_ARGS__); \
    ret name(__VA_ARGS__);

#define FUNC_DEF_NOCUDA(ret, name, ...) \
    ret name##_cpu(__VA_ARGS__); \
    ret name(__VA_ARGS__);

#define FUNC_IMPL_DISPATCH(ret, name, ...) ret name(__VA_ARGS__)
#if (CUDA_ENABLED==1)
#define FUNC_IMPL_CPU(ret, name, ...) ret name##_cpu(__VA_ARGS__)
#else
#define FUNC_IMPL_CPU(ret, name, ...) ret name##_cuda(__VA_ARGS__) { throw std::logic_error("Not compiled with CUDA support."); } \
    ret name##_cpu(__VA_ARGS__)
#endif
#define FUNC_IMPL_CUDA(ret, name, ...) ret name##_cuda(__VA_ARGS__)

inline bool is_cuda(const torch::Tensor& x) {
    return x.device().is_cuda();
}

/* Sparse GEMV */
FUNC_DEF(torch::Tensor,
         spgemv_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
         torch::Tensor x);

FUNC_DEF(std::vector<torch::Tensor>,
         spgemv_backward,
         torch::Tensor grad_z, int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
         torch::Tensor x);

/* Sparse GEMM */
FUNC_DEF(std::vector<torch::Tensor>,
         spgemm_forward,
         int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr);

FUNC_DEF(std::vector<torch::Tensor>,
         spgemm_backward,
         torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
         int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr);

/* CSR Transpose */
FUNC_DEF(std::vector<torch::Tensor>,
         csr_transpose_forward,
         int A_rows, int A_columns,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

FUNC_DEF(torch::Tensor,
         csr_transpose_backward,
         torch::Tensor grad_At, torch::Tensor At_to_A_idx);

/* Sparse linear combination */
FUNC_DEF(std::vector<torch::Tensor>,
         splincomb_forward,
         int rows, int cols,
         torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr);

FUNC_DEF(std::vector<torch::Tensor>,
         splincomb_backward,
         int rows, int cols,
         torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr,
         torch::Tensor grad_C_data, torch::Tensor C_indices, torch::Tensor C_indptr);

/* Sparse times dense */
FUNC_DEF(torch::Tensor,
         spdmm_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         torch::Tensor B);

FUNC_DEF(std::vector<torch::Tensor>,
         spdmm_backward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         torch::Tensor B, torch::Tensor grad_C);

/* Sparse triangular solve */
FUNC_DEF(torch::Tensor,
         sptrsv_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         bool lower, bool unit, torch::Tensor b);

FUNC_DEF(std::vector<torch::Tensor>,
         sptrsv_backward,
         torch::Tensor grad_x, torch::Tensor x,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
         bool lower, bool unit, torch::Tensor b);

/* LU Factorization */
FUNC_DEF(std::vector<torch::Tensor>,
         splu,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

FUNC_DEF(std::vector<torch::Tensor>,
         spsolve_backward,
         torch::Tensor grad_x, torch::Tensor x,
         int A_rows, int A_cols,
         torch::Tensor Mt_data, torch::Tensor Mt_indices, torch::Tensor Mt_indptr,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

/* To Dense */
FUNC_DEF(torch::Tensor,
         csr_to_dense_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

FUNC_DEF(torch::Tensor,
         csr_to_dense_backward,
         torch::Tensor grad_Ad,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

/* Row sum */
FUNC_DEF(torch::Tensor,
         csr_row_sum_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

FUNC_DEF(torch::Tensor,
         csr_row_sum_backward,
         torch::Tensor grad_x,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

/* CSR Extract diagonal */
FUNC_DEF(torch::Tensor,
         csr_extract_diagonal_forward,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

FUNC_DEF(torch::Tensor,
         csr_extract_diagonal_backward,
         torch::Tensor grad_x,
         int A_rows, int A_cols,
         torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr);

#endif
