#ifndef SPARSE_CSR_H_
#define SPARSE_CSR_H_

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
FUNC_DEF(std::vector<torch::Tensor>,
         spgemv_forward,
         int A_rows, int A_cols, torch::Tensor alpha,
         torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
         torch::Tensor x, torch::Tensor beta, torch::Tensor y);

FUNC_DEF(std::vector<torch::Tensor>,
         spgemv_backward,
         torch::Tensor grad_z, int A_rows, int A_cols, torch::Tensor alpha,
         torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
         torch::Tensor x, torch::Tensor beta, torch::Tensor y);

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

#endif
