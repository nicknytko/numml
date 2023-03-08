#include <torch/extension.h>
#include <map>
#include <tuple>
#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <iterator>
#include <tuple>
#include <iostream>

#include "cuda_common.cuh"
#include "sparse_csr.hpp"

/* Sparse GEMV */

/**
 * Computes the sparse matvec Ax.
 * Indexed on entries of the output.
 */
template <typename scalar_t>
__global__ void spgemv_forward_cuda_kernel_matvec(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_ind,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_rowptr,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> z_out) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    float z_i = 0.f;
    for (int64_t i = A_rowptr[row]; i < A_rowptr[row + 1]; i++) {
        int col = A_col_ind[i];
        z_i += A_data[i] * x[col];
    }
    z_out[row] = z_i;
}

FUNC_IMPL_CUDA(torch::Tensor,
               spgemv_forward,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
               torch::Tensor x) {

    auto options = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor Ax = torch::empty({A_rows}, options);
    const int threads = A_rows;
    const dim3 blocks((threads + threads_per_block - 1) / threads_per_block, 1);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_forward_cuda", ([&] {
        spgemv_forward_cuda_kernel_matvec<scalar_t><<<blocks, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols,
            tensor_acc(A_data, scalar_t), tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t),
            tensor_acc(x, scalar_t), tensor_acc(Ax, scalar_t));
    }));
    cuda_check_kernel_launch_err();

    return Ax;
}

/**
 * Computes the masked outer product like
 * C = alpha * ab^T (*) mask(C),
 * where it is assumed that the CSR structure of C is known and only nonzeros are to be filled in.
 * Indexed on rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_masked_outerproduct(
    const int64_t C_rows, const int64_t C_cols, scalar_t alpha,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> a,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> b,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data) {

    /* grad_A = alpha * outer(grad_z, x) (*) mask(A) */

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= C_rows) {
        return;
    }

    for (int64_t i = C_indptr[row]; i < C_indptr[row + 1]; i++) {
        const int64_t col = C_indices[i];
        C_data[i] = alpha * a[row] * b[col];
    }
}

/**
 * Computes the gradient of Ax wrt x in the spgemv product.
 * Indexed on entries of A.
 *
 * Based on the implementation in
 * "Atomic reduction based sparse matrix-transpose vector multiplication on GPUs",
 * Y. Tao, Y. Deng, S. Mu, ICPADS (2014)
 */
template <typename scalar_t>
__global__ void spgemv_backward_cuda_kernel_grad_x_atomic(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_z,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_col_ind,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_rowptr,
    scalar_t* __restrict__ grad_x) {

    /* Compute grad_x = A^T grad_z */

    const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= A_rows) {
        return;
    }

    for (int64_t k_i = A_rowptr[k]; k_i < A_rowptr[k + 1]; k_i++) {
        const int64_t i = A_col_ind[k_i];
        atomicAdd(grad_x + i, A_data[k_i] * grad_z[k]);
    }
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemv_backward,
               torch::Tensor grad_z, int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
               torch::Tensor x) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* compute grad_A = grad_z x^T (*) mask(A) */
    torch::Tensor grad_A = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_backward_cuda", ([&] {
        cuda_kernel_masked_outerproduct<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols, 1.,
            tensor_acc(grad_z, scalar_t), tensor_acc(x, scalar_t),
            tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t), tensor_acc(grad_A, scalar_t));
    }));
    cuda_check_kernel_launch_err();

    /* compute grad_x = A^T grad_z */
    torch::Tensor grad_x;
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_backward_cuda", ([&] {
        /* We'll create a temporary array to store outputs to and simplify things */
        scalar_t* grad_x_ary = nullptr;
        cudaMalloc(&grad_x_ary, A_cols * sizeof(scalar_t));
        cudaMemsetAsync(grad_x_ary, 0, A_cols * sizeof(scalar_t), main_stream);

        const int grad_x_threads = A_rows;
        const dim3 grad_x_blocks((grad_x_threads + threads_per_block - 1) / threads_per_block, 1);
        spgemv_backward_cuda_kernel_grad_x_atomic<scalar_t><<<grad_x_blocks, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols,
            tensor_acc(grad_z, scalar_t),
            tensor_acc(A_data, scalar_t), tensor_acc(A_col_ind, int64_t), tensor_acc(A_rowptr, int64_t),
            grad_x_ary);
        cuda_check_kernel_launch_err();

        /* Array blob -> torch tensor.  Torch will handle deallocation if we pass it a destructor. */
        const auto scalar_tens_opts = torch::TensorOptions()
            .dtype(A_data.dtype())
            .device(A_data.device().type(), A_data.device().index());
        grad_x = torch::from_blob(grad_x_ary, { static_cast<int64_t>(A_cols) }, cudaFree, scalar_tens_opts);
    }));

    return {grad_A, grad_x};
}

/**
 * Find the number of nonzero entries per column using atomic operations.
 * Indexed on rows of A.
 */
__global__ void cuda_kernel_csr_nnz_per_col(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    int64_t* __restrict__ A_col_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    for (int64_t j = A_indptr[i]; j < A_indptr[i + 1]; j++) {
        const int64_t col = A_indices[j];
        atomicAdd((unsigned long long int*) (A_col_nnz + col), 1ull);
    }
}

template <typename scalar_t>
__global__ void cuda_kernel_csr_transpose_accumulate(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> At_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_to_A_idx) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_cols) {
        return;
    }
    int64_t dest_ptr = At_indptr[i];

    for (int64_t row = 0; row < A_rows; row++) {
        for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
            int64_t column = A_indices[row_i];
            if (column != i) {
                continue;
            }

            At_indices[dest_ptr] = row;
            At_data[dest_ptr] = A_data[row_i];
            At_to_A_idx[dest_ptr] = row_i;

            dest_ptr ++;
        }
    }
}

/* CSR Transpose */
FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               csr_transpose_forward,
               int A_rows, int A_columns,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    /* Based on the serial implementation from Scipy:
       https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L380 */

    const int64_t nnz = A_indptr[A_rows].item<int>();

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor At_data = torch::empty(nnz, scalar_tens_opts);
    torch::Tensor At_indptr;
    torch::Tensor At_indices = torch::empty(nnz, int_tens_opts);
    int64_t* At_indptr_raw;
    int64_t* At_indptr_tmp;
    cuda_check_err(cudaMalloc(&At_indptr_tmp, A_columns * sizeof(int64_t)));
    cuda_check_err(cudaMalloc(&At_indptr_raw, (A_columns + 1) * sizeof(int64_t)));
    cuda_check_err(cudaMemsetAsync(At_indptr_tmp, 0, A_columns * sizeof(int64_t), main_stream));
    cuda_check_err(cudaMemsetAsync(At_indptr_raw, 0, sizeof(int64_t), main_stream)); /* Zero out first entry. */

    /* Compute number of nonzeros per column of A */
    cuda_kernel_csr_nnz_per_col<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows, A_columns,
        tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t), At_indptr_tmp);
    cuda_check_kernel_launch_err();

    /* Now, compute the cumulative sum of nnz to get starting rowptrs of A^T */
    cub_cumsum(At_indptr_tmp, At_indptr_raw + 1, A_columns, main_stream);
    cuda_check_err(cudaFreeAsync(At_indptr_tmp, main_stream));
    At_indptr = torch::from_blob(At_indptr_raw, { static_cast<int64_t>(A_columns + 1) }, cudaFree, int_tens_opts);

    /* Move data values into their correct spots */
    torch::Tensor At_to_A_idx = torch::empty(nnz, int_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_transpose_forward_cuda", ([&] {
        cuda_kernel_csr_transpose_accumulate<<<(A_columns + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_columns,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
            tensor_acc(At_data, scalar_t), tensor_acc(At_indices, int64_t), tensor_acc(At_indptr, int64_t),
            tensor_acc(At_to_A_idx, int64_t));
    }));
    cuda_check_kernel_launch_err();

    return {At_data, At_indices, At_indptr, At_to_A_idx};
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_transpose_backward,
               torch::Tensor grad_At, torch::Tensor At_to_A_idx) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A = torch::empty_like(grad_At);

    AT_DISPATCH_FLOATING_TYPES(grad_At.type(), "csr_transpose_backward_cuda", ([&] {
        cuda_kernel_tensor_permute<<<(grad_At.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            grad_At.size(0), tensor_acc(grad_At, scalar_t), tensor_acc(grad_A, scalar_t), tensor_acc(At_to_A_idx, int64_t), false);
    }));
    cuda_check_kernel_launch_err();

    return grad_A;
}

/**
 * Computes number of nonzeros per row when performing a linear combination
 * of two sparse matrices.
 *
 * Indexed on rows of the output.
 */
__global__ void cuda_kernel_splincomb_nnz_per_row(int rows, int cols,
                                                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
                                                  const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
                                                  torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> nnz_per_row) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int64_t nnz = 0;

    int64_t i_A = A_indptr[row];
    int64_t i_B = B_indptr[row];

    int64_t end_A = A_indptr[row + 1];
    int64_t end_B = B_indptr[row + 1];

    /* Merge the row of A and B */
    while (i_A < end_A && i_B < end_B) {
        int64_t col_A = A_indices[i_A];
        int64_t col_B = B_indices[i_B];

        if (col_A < col_B) {
            nnz++;
            i_A++;
        } else if (col_A > col_B) {
            nnz++;
            i_B++;
        } else { /* we hit the same row-column pair in both matrices */
            nnz++;
            i_A++;
            i_B++;
        }
    }

    /* Exhausted shared indices, now we add the rest of the row of A or B */
    while (i_A < end_A) {
        nnz++;
        i_A++;
    }
    while (i_B < end_B) {
        nnz++;
        i_B++;
    }

    nnz_per_row[row] = nnz;
}

/**
 * Computes the sparse linear combination of two matrices.
 *
 * Indexed on rows of the output.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splincomb(int rows, int cols,
                                      scalar_t alpha, scalar_t beta,
                                      torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
                                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                      torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
                                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
                                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
                                      const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> cum_nnz_per_row,
                                      torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
                                      torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices,
                                      torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    /* Set C_indptr */
    int64_t C_ptr = (row > 0 ? cum_nnz_per_row[row-1] : 0);
    C_indptr[row] = C_ptr;
    if (row == rows-1) {
        C_indptr[rows] = cum_nnz_per_row[rows - 1];
    }

    int64_t i_A = A_indptr[row];
    int64_t i_B = B_indptr[row];

    int64_t end_A = A_indptr[row + 1];
    int64_t end_B = B_indptr[row + 1];

    /* Merge the row of A and B */
    while (i_A < end_A && i_B < end_B) {
        int64_t col_A = A_indices[i_A];
        int64_t col_B = B_indices[i_B];

        if (col_A < col_B) {
            C_data[C_ptr] = alpha * A_data[i_A];
            C_indices[C_ptr] = col_A;
            C_ptr++;
            i_A++;
        } else if (col_A > col_B) {
            C_data[C_ptr] = beta * B_data[i_B];
            C_indices[C_ptr] = col_B;
            C_ptr++;
            i_B++;
        } else { /* we hit the same row-column pair in both matrices */
            C_data[C_ptr] = alpha * A_data[i_A] + beta * B_data[i_B];
            C_indices[C_ptr] = col_A;
            C_ptr++;
            i_A++;
            i_B++;
        }
    }

    /* Exhausted shared indices, now we add the rest of the row of A or B */
    while (i_A < end_A) {
        C_data[C_ptr] = alpha * A_data[i_A];
        C_indices[C_ptr] = A_indices[i_A];
        C_ptr++;
        i_A++;
    }
    while (i_B < end_B) {
        C_data[C_ptr] = beta * B_data[i_B];
        C_indices[C_ptr] = B_indices[i_B];
        C_ptr++;
        i_B++;
    }
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               splincomb_forward,
               int rows, int cols,
               torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Start by computing the number of elements per row. */
    torch::Tensor nnz_per_row = torch::empty({rows}, int_tens_opts);
    cuda_kernel_splincomb_nnz_per_row<<<(rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        rows, cols,
        tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
        tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
        tensor_acc(nnz_per_row, int64_t));
    cuda_check_kernel_launch_err();
    torch::Tensor cum_nnz_per_row = nnz_per_row.cumsum(0);
    const int64_t total_nnz = cum_nnz_per_row[rows - 1].item<int64_t>();

    /* Now combine both matrices */
    torch::Tensor C_data = torch::empty({total_nnz}, scalar_tens_opts);
    torch::Tensor C_indices = torch::empty({total_nnz}, int_tens_opts);
    torch::Tensor C_indptr = torch::empty({rows + 1}, int_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splincomb_forward_cuda", ([&] {
        cuda_kernel_splincomb<<<(rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            rows, cols, alpha.item<scalar_t>(), beta.item<scalar_t>(),
            tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
            tensor_acc(B_data, scalar_t), tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
            tensor_acc(cum_nnz_per_row, int64_t), tensor_acc(C_data, scalar_t), tensor_acc(C_indices, int64_t), tensor_acc(C_indptr, int64_t));
    }));
    cuda_check_kernel_launch_err();

    return {C_data, C_indices, C_indptr};
}

template <typename scalar_t>
__global__ void cuda_kernel_splincomb_backward(int rows, int cols,
                                               scalar_t scalar,
                                               const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> gC_data,
                                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices,
                                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr,
                                               torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> gA_data,
                                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                               const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }

    int64_t A_i = A_indptr[row];
    int64_t C_i = C_indptr[row];

    int64_t A_end = A_indptr[row + 1];
    int64_t C_end = C_indptr[row + 1];

    while (A_i < A_end && C_i < C_end) {
        int64_t A_col = A_indices[A_i];
        int64_t C_col = C_indices[C_i];

        if (A_col < C_col) {
            A_i++;
        } else if (A_col > C_col) {
            C_i++;
        } else {
            gA_data[A_i] = gC_data[C_i] * scalar;
            A_i++;
            C_i++;
        }
    }
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               splincomb_backward,
               int rows, int cols,
               torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr,
               torch::Tensor grad_C_data, torch::Tensor C_indices, torch::Tensor C_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* grad_A = (alpha * grad_c) (*) mask(A) */
    torch::Tensor grad_A = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splincomb_backward_cuda", ([&] {
        cuda_kernel_splincomb_backward<<<(rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            rows, cols, alpha.item<scalar_t>(),
            tensor_acc(grad_C_data, scalar_t), tensor_acc(C_indices, int64_t), tensor_acc(C_indptr, int64_t),
            tensor_acc(grad_A, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));
    cuda_check_kernel_launch_err();

    /* grad_B = (beta * grad_c) (*) mask(A) */
    torch::Tensor grad_B = torch::empty_like(B_data);
    AT_DISPATCH_FLOATING_TYPES(B_data.type(), "splincomb_backward_cuda", ([&] {
        cuda_kernel_splincomb_backward<<<(rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            rows, cols, beta.item<scalar_t>(),
            tensor_acc(grad_C_data, scalar_t), tensor_acc(C_indices, int64_t), tensor_acc(C_indptr, int64_t),
            tensor_acc(grad_B, scalar_t), tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t));
    }));
    cuda_check_kernel_launch_err();

    return {grad_A, grad_B};
}

/**
 * Compute the inner product between a sparse CSR matrix and dense matrix.
 * Indexed on blocks of the output.
 */
template <typename scalar_t>
__global__ void cuda_kernel_spdmm_forward(int A_rows, int A_cols, int B_rows, int B_cols,
                                          const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
                                          const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                          const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                          const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B,
                                          torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> C) {

    const int64_t c_i = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t c_j = blockIdx.y * blockDim.y + threadIdx.y;

    const int64_t C_rows = A_rows;
    const int64_t C_cols = B_cols;

    if (c_i >= C_rows || c_j >= C_cols) {
        return;
    }

    scalar_t cij = 0.;
    for (int64_t i_i = A_indptr[c_i]; i_i < A_indptr[c_i + 1]; i_i++) {
        const int64_t k = A_indices[i_i];
        cij += A_data[i_i] * B[k][c_j];
    }
    C[c_i][c_j] = cij;
}

FUNC_IMPL_CUDA(torch::Tensor,
               spdmm_forward,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               torch::Tensor B) {

    const int64_t B_rows = B.size(0);
    const int64_t B_cols = B.size(1);
    const int64_t C_rows = A_rows;
    const int64_t C_cols = B_cols;

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor C = torch::empty({C_rows, C_cols}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spdmm_forward_cuda", ([&] {
        const dim3 blocks((C_rows + threads_per_block_2d - 1) / threads_per_block_2d,
                          (C_cols + threads_per_block_2d - 1) / threads_per_block_2d,
                          1);
        const dim3 threads(threads_per_block_2d, threads_per_block_2d, 1);
        cuda_kernel_spdmm_forward<scalar_t><<<blocks, threads, 0, main_stream>>>(
            A_rows, A_cols, B_rows, B_cols,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
            tensor_acc_3(B, 2, scalar_t), tensor_acc_3(C, 2, scalar_t));
    }));
    cuda_check_kernel_launch_err();

    return C;
}

/**
 * Computes grad_A = (grad_C * B^T) (*) mask(A)
 *
 * Indexed on rows of A.
 */
template <typename scalar_t>
__global__ void cuda_kernel_spdmm_backward_masked_A(const int64_t A_rows, const int64_t A_cols,
                                                    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_A_data,
                                                    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                                    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                                    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> B,
                                                    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_C) {

    int64_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= grad_A_data.size(0)) {
        return;
    }

    /* Find row and column index */
    int64_t col = A_indices[out_idx];
    int64_t row = 0, row_l = 0, row_h = A_rows - 1;
    while (true) {
        int64_t row_m = (row_l + row_h)/2;

        if (out_idx >= A_indptr[row_m] && out_idx < A_indptr[row_m+1]) {
            row = row_m;
            break;
        } else if (out_idx < A_indptr[row_m]) {
            row_h = row_m - 1;
        } else {
            row_l = row_m + 1;
        }
    }
    __syncthreads();

    scalar_t a_ij = 0.;
    for (int64_t k = 0; k < B.size(1); k++) {
        a_ij += grad_C[row][k] * B[col][k];
    }
    grad_A_data[out_idx] = a_ij;
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spdmm_backward,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               torch::Tensor B, torch::Tensor grad_C) {

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    // auto start = std::chrono::steady_clock::now();

    /* grad_A = (grad_C * B^T) (*) mask(A) */
    torch::Tensor grad_A;
    if (A_data.requires_grad()) {
        grad_A = torch::empty_like(A_data);
        AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spdmm_backward_cuda", ([&] {
            cuda_kernel_spdmm_backward_masked_A<scalar_t><<<(A_data.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols,
                tensor_acc(grad_A, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
                tensor_acc_3(B, 2, scalar_t), tensor_acc_3(grad_C, 2, scalar_t));
        }));
        cuda_check_kernel_launch_err();
    }

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // std::cerr << "spdmm_backward grad_A " << diff.count() << std::endl;

    /* grad_B = (A^T * grad_C) */
    torch::Tensor grad_B;

    // start = std::chrono::steady_clock::now();

    if (B.requires_grad()) {
        grad_B = torch::empty({A_cols, grad_C.size(1)}, scalar_tens_opts);
        auto At = csr_transpose_forward_cuda(A_rows, A_cols, A_data, A_indices, A_indptr);
        AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spdmm_backward_cuda", ([&] {
            const dim3 blocks((A_cols + threads_per_block_2d - 1) / threads_per_block_2d,
                              (grad_C.size(1) + threads_per_block_2d - 1) / threads_per_block_2d,
                              1);
            const dim3 threads(threads_per_block_2d, threads_per_block_2d, 1);
            cuda_kernel_spdmm_forward<scalar_t><<<blocks, threads, 0, main_stream>>>(
                A_cols, A_rows, grad_C.size(0), grad_C.size(1),
                tensor_acc(At[0], scalar_t), tensor_acc(At[1], int64_t), tensor_acc(At[2], int64_t),
                tensor_acc_3(grad_C, 2, scalar_t), tensor_acc_3(grad_B, 2, scalar_t));
        }));
        cuda_check_kernel_launch_err();
    }

    // end = std::chrono::steady_clock::now();
    // diff = end - start;
    // std::cerr << "spdmm_backward grad_A " << diff.count() << std::endl;

    return {grad_A, grad_B};
}

/**
 * Computes the lower triangular solve of
 * Lx = b
 * based on the write-first CapelliniSpTRSV algorithm in
 * "CapelliniSpTRSV: A Thread-Level Synchronization-Free Sparse Triangular Solve on GPUs", Su et al
 *
 * Indexed on rows of L.
 */
template <typename scalar_t>
__global__ void cuda_kernel_sptrsv_forward_lower(const int64_t A_rows, const int64_t A_cols,
                                                 const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
                                                 const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                                 const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                                 const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> b,
                                                 volatile double* __restrict__ x,
                                                 bool unit,
                                                 volatile bool* __restrict__ value_available) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    double acc = 0.;
    int64_t j = A_indptr[i];
    int64_t col = A_indices[j];
    while (j < A_indptr[i+1]) {
        /* If we hit the diagonal then we're done and can update our entry in the output */
        if (col == i) {
            if (unit) {
                x[i] = static_cast<double>(b[i]) - acc;
            } else {
                x[i] = (static_cast<double>(b[i]) - acc) / static_cast<double>(A_data[j]);
            }
            __threadfence();
            value_available[i] = true;
            break;
        }

        /* Implicitly busywait until we can accumulate the entire row. */
        while (value_available[col] && col != i) {
            acc += static_cast<double>(A_data[j]) * x[col];
            ++j;
            col = A_indices[j];
        }
    }
}


template <typename scalar_t>
__global__ void cuda_kernel_sptrsv_forward_upper(const int64_t A_rows, const int64_t A_cols,
                                                 const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
                                                 const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
                                                 const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
                                                 const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> b,
                                                 volatile double* __restrict__ x,
                                                 bool unit,
                                                 volatile bool* __restrict__ value_available) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    double acc = 0.;
    int64_t j = A_indptr[i+1] - 1;
    int64_t col = A_indices[j];
    while (j >= A_indptr[i]) {
        /* If we hit the diagonal then we're done and can update our entry in the output */
        if (col == i) {
            if (unit) {
                x[i] = static_cast<double>(b[i]) - acc;
            } else {
                x[i] = (static_cast<double>(b[i]) - acc) / static_cast<double>(A_data[j]);
            }
            __threadfence();
            value_available[i] = true;
            break;
        }

        /* Implicitly busywait until we can accumulate the entire row. */
        while (value_available[col] && col != i) {
            acc += static_cast<double>(A_data[j]) * x[col];
            --j;
            col = A_indices[j];
        }
    }
}

FUNC_IMPL_CUDA(torch::Tensor,
              sptrsv_forward,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              bool lower, bool unit, torch::Tensor b) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(A_data.device().type(), A_data.device().index());
    torch::Tensor x_dbl;

    /* We allocate these as raw bool and double arrays so that we can mark them as volatile;
       torch doesn't have a native way to do this for tensor accessors */
    bool* value_available;
    cuda_check_err(cudaMalloc(&value_available, sizeof(bool) * A_rows));
    cuda_check_err(cudaMemsetAsync(value_available, 0, sizeof(bool) * A_rows, main_stream));

    double* x_raw;
    cuda_check_err(cudaMalloc(&x_raw, sizeof(double) * A_rows));

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "sptrsv_forward_cuda", ([&] {
        if (lower) {
            cuda_kernel_sptrsv_forward_lower<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
                tensor_acc(b, scalar_t), x_raw, unit, value_available);
        } else {
            cuda_kernel_sptrsv_forward_upper<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
                tensor_acc(b, scalar_t), x_raw, unit, value_available);
        }
    }));
    cuda_check_kernel_launch_err();

    cuda_check_err(cudaFreeAsync(value_available, main_stream));
    x_dbl = torch::from_blob(x_raw, { static_cast<int64_t>(A_rows) }, cudaFree, options);
    return x_dbl.to(A_data.dtype());
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               sptrsv_backward,
               torch::Tensor grad_x, torch::Tensor x,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               bool lower, bool unit, torch::Tensor b) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Compute grad_b = A^{-T} grad_c */
    auto At = csr_transpose_forward_cuda(A_rows, A_cols, A_data, A_indices, A_indptr);
    torch::Tensor At_data = At[0];
    torch::Tensor At_indices = At[1];
    torch::Tensor At_indptr = At[2];

    torch::Tensor grad_b = sptrsv_forward_cuda(A_rows, A_cols, At_data, At_indices, At_indptr, !lower, unit, grad_x);

    /* Compute grad_A = -grad_b x^T (*) mask(A) */
    torch::Tensor grad_A_data = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "sptrsv_backward_cuda", ([&] {
        cuda_kernel_masked_outerproduct<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols, -1., tensor_acc(grad_b, scalar_t), tensor_acc(x, scalar_t),
            tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t), tensor_acc(grad_A_data, scalar_t));
    }));
    cuda_check_kernel_launch_err();

    return {grad_A_data, grad_b};
}

/**
 * Compute the number of nonzero entries in each column of the LU factorization.
 * Finds fill-in using a depth-first traversal of the matrix.  Based on
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs",  Gaihre A, Li X, Liu H.
 *
 * This should be run with one block of some predetermined fixed size.  This is
 * run with less threads than overall columns due to memory constraints.
 */
__global__ void cuda_kernel_splu_symbolic_fact_trav_nnz(
    const int64_t A_rows, const int64_t A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    int64_t* __restrict__ vert_fill,
    int64_t* __restrict__ vert_queue,
    bool* __restrict__ vert_mask,
    int64_t* __restrict__ As_nnz) {

    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = thread_idx;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {
        /* Zero out bitmap of visited nodes */
        for (int64_t i = 0; i < A_cols; i++) {
            vert_fill[thread_idx * A_cols + i] = 0;
            vert_mask[thread_idx * A_cols + i] = false;
        }

        /* Set fill array */
        for (int64_t v_i = A_indptr[row]; v_i < A_indptr[row + 1]; v_i++) {
            const int64_t v = A_indices[v_i];
            vert_fill[thread_idx * A_cols + v] = row;
            vert_mask[thread_idx * A_cols + v] = true;
        }
        __syncthreads();

        /* Loop over "threshold" */
        for (int64_t t = 0; t < row; t++) {
            if (vert_fill[thread_idx * A_rows + t] != row) {
                continue;
            }

            int64_t queue_start = 0;
            int64_t queue_end = 1;
            vert_queue[thread_idx * A_rows] = t;

            while (queue_start != queue_end) {
                const int64_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
                queue_start++;

                for (int64_t w_i = A_indptr[u]; w_i < A_indptr[u + 1]; w_i++) {
                    const int64_t w = A_indices[w_i];
                    if (vert_fill[thread_idx * A_rows + w] < row) {
                        vert_fill[thread_idx * A_rows + w] = row;
                        if (w > t) {
                            vert_mask[thread_idx * A_rows + w] = true;
                        } else {
                            vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                            queue_end++;
                        }
                    }
                }
            }
        }
        __syncthreads();

        /* Count number of nonzeros in L and U in the current column */
        int64_t As_nnz_row = 0;
        for (int64_t i = 0; i < A_cols; i++) {
            if (vert_mask[thread_idx * A_rows + i]) {
                As_nnz_row++;
                vert_mask[thread_idx * A_rows + i] = false;
            }
        }
        As_nnz[row] = As_nnz_row;
        row += blockDim.x * gridDim.x;
    }
}

/**
 * Given number of nonzero fill-ins in the column LU factorization, populate
 * row indices and data entries of the symbolic factorization.  Based on
 * "GSoFa: Scalable Sparse Symbolic LU Factorization on GPUs",  Gaihre A, Li X, Liu H.
 *
 * This should be run with one block of some predetermined fixed size.  This is
 * run with less threads than overall columns due to memory constraints.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_symbolic_fact_trav_populate(
    const int64_t A_rows, const int64_t A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    int64_t* __restrict__ vert_fill,
    int64_t* __restrict__ vert_queue,
    bool* __restrict__ vert_mask,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> As_row_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> As_col_indices,
    const int64_t* __restrict__ As_row_indptr) {

    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t row = thread_idx;

    /* We'll round robin over the columns to save memory */
    while (row < A_rows) {
        /* Zero out bitmap of visited nodes */
        for (int64_t i = 0; i < A_rows; i++) {
            vert_fill[thread_idx * A_rows + i] = 0;
            vert_mask[thread_idx * A_rows + i] = false;
        }

        /* Set fill array */
        for (int64_t v_i = A_indptr[row]; v_i < A_indptr[row + 1]; v_i++) {
            const int64_t v = A_indices[v_i];
            vert_fill[thread_idx * A_rows + v] = row;
            vert_mask[thread_idx * A_rows + v] = true;
        }
        __syncthreads();

        /* Loop over "threshold" */
        for (int64_t t = 0; t < row; t++) {
            if (vert_fill[thread_idx * A_rows + t] != row) {
                continue;
            }

            int64_t queue_start = 0;
            int64_t queue_end = 1;
            vert_queue[thread_idx * A_rows] = t;

            while (queue_start != queue_end) {
                const int64_t u = vert_queue[thread_idx * A_rows + (queue_start % A_rows)];
                queue_start++;

                for (int64_t w_i = A_indptr[u]; w_i < A_indptr[u + 1]; w_i++) {
                    const int64_t w = A_indices[w_i];
                    if (vert_fill[thread_idx * A_rows + w] < row) {
                        vert_fill[thread_idx * A_rows + w] = row;
                        if (w > t) {
                            vert_mask[thread_idx * A_rows + w] = true;
                        } else {
                            vert_queue[thread_idx * A_rows + (queue_end % A_rows)] = w;
                            queue_end++;
                        }
                    }
                }
            }
        }
        __syncthreads();

        /* Insert row indices and nonzero values of At_data.
           This is essentially a union of the two columns, where entries in As *only* are explicitly zero. */

        int64_t As_ptr = 0; /* Current entry in vert_visited array */
        int64_t A_ptr = A_indptr[row]; /* Current index in original A */
        int64_t As_out_ptr = As_row_indptr[row]; /* Current index in output As */

        const int64_t As_end = A_cols;
        const int64_t A_end = A_indptr[row + 1];

        while (As_ptr < As_end && A_ptr < A_end) {
            /* Make sure we actually are at a nonzero of As */
            while (!vert_mask[thread_idx * A_rows + As_ptr]) {
                As_ptr++;
            }

            const int64_t As_col = As_ptr;
            const int64_t A_col = A_indices[A_ptr];
            if (As_col < A_col) {
                As_row_data[As_out_ptr] = 0.;
                As_col_indices[As_out_ptr] = As_col;

                As_ptr++;
                As_out_ptr++;
            } else if (As_col > A_col) {
                /* This is probably unlikely, since A is a subset of As..?
                   Nonetheless, let's add it here just in case. */
                As_row_data[As_out_ptr] = A_data[A_ptr];
                As_col_indices[As_out_ptr] = A_col;

                A_ptr++;
                As_out_ptr++;
            } else { /* As_col == A_col */
                As_row_data[As_out_ptr] = A_data[A_ptr];
                As_col_indices[As_out_ptr] = A_col;

                A_ptr++;
                As_ptr++;
                As_out_ptr++;
            }
        }
        /* Finish off with rest of As entries */
        for (; As_ptr < As_end; As_ptr++) {
            if (vert_mask[thread_idx * A_rows + As_ptr]) {
                As_row_data[As_out_ptr] = 0.;
                As_col_indices[As_out_ptr] = As_ptr;
                As_out_ptr++;
            }
        }

        row += blockDim.x * gridDim.x;
    }
}

/**
 * Count the number of upper-triangular nonzeros for each column of a CSC matrix.
 * This is inclusive of the main diagonal.
 *
 * Indexed on columns of A.
 */
__global__ void cuda_kernel_count_U_nnz(
    const int64_t A_rows, const int64_t A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> At_indptr,
    int64_t* __restrict__ U_col_nnz) {

    const int64_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= A_cols) {
        return;
    }

    int64_t nnz = 0;
    for (int64_t i_i = At_indptr[j]; i_i < At_indptr[j + 1]; i_i++) {
        const int64_t i = At_indices[i_i];
        if (i <= j) {
            nnz++;
        }
    }

    U_col_nnz[j] = nnz;
}

/**
 * The sparse numeric LU factorization from SFLU:
 * "SFLU: Synchronization-Free Sparse LU Factorization for Fast Circuit Simulation on GPUs", J. Zhao, Y. Luo, Z. Jin, Z. Zhou.
 *
 * Indexed on columns of As, where As is given in CSC format and has fill-ins represented by explicit zeros.
 */
template <typename scalar_t>
__global__ void cuda_kernel_splu_numeric_sflu(
        const int64_t A_rows, const int64_t A_cols,
        torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> As_col_data,
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> As_col_indices,
        const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> As_col_indptr,
        volatile int64_t* __restrict__ degree) {

    const int64_t k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= A_cols) {
        return;
    }

    int64_t diag_idx;
    const int64_t col_end = As_col_indptr[k + 1];
    for (int64_t i_i = As_col_indptr[k]; i_i < col_end; i_i++) {
        const int64_t i = As_col_indices[i_i];
        if (i == k) {
            /* Stop once we get to the diagonal. */
            diag_idx = i_i;
            break;
        }

        /* Busy wait until intermediate results are ready */
        while (degree[i] > 0);

        /* Left-looking product */
        for (int64_t j_i = i_i + 1; j_i < col_end; j_i++) {
            const int64_t j = As_col_indices[j_i];
            const int64_t A_ji_i = kernel_indices_binsearch(As_col_indptr[i], As_col_indptr[i + 1] - 1, j, As_col_indices);
            if (A_ji_i == -1) {
                continue;
            }
            const scalar_t A_ji = As_col_data[A_ji_i];
            const scalar_t A_ik = As_col_data[i_i];

            /* A_{jk} \gets A_{jk} - A_{ji} A_{ik} */
            As_col_data[j_i] -= A_ji * A_ik;
        }

        __threadfence();
        degree[k]--;
    }

    /* Divide column of L by diagonal entry of U */
    const scalar_t A_kk = As_col_data[diag_idx];
    for (int64_t i = diag_idx + 1; i < As_col_indptr[k + 1]; i++) {
        As_col_data[i] /= A_kk;
    }

    /* Complete the factorization and update column degree */
    __threadfence();
    degree[k]--;
}

/**
 * Sparse LU Factorization, using a left-looking algorithm on the columns of A.  Based on
 * the symbolic factorization from Rose, Tarjan's fill2 and numeric factorization in SFLU.
 */
FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               splu,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* First, perform the symbolic factorization to determine the sparsity pattern of the filled-in LU factorization
       of A, which we will hereby denote by As.  Note that mask(As) \superset mask(A). */
    const int64_t num_threads_symb = 32;
    const int64_t num_blocks_symb = 8;
    const int64_t total_threads_symb = num_threads_symb * num_blocks_symb;
    int64_t* vert_fill;
    int64_t* vert_queue;
    bool* vert_mask;
    int64_t* As_row_nnz;
    int64_t* As_row_indptr_raw;
    int64_t* U_col_nnz;

    cuda_check_err(cudaMalloc(&vert_fill, sizeof(int64_t) * total_threads_symb * A_rows));
    cuda_check_err(cudaMalloc(&vert_queue, sizeof(int64_t) * total_threads_symb * A_rows));
    cuda_check_err(cudaMalloc(&vert_mask, sizeof(bool) * total_threads_symb * A_rows));
    cuda_check_err(cudaMalloc(&As_row_nnz, sizeof(int64_t) * A_rows));
    cuda_check_err(cudaMalloc(&As_row_indptr_raw, sizeof(int64_t) * (A_rows + 1)));
    cuda_check_err(cudaMalloc(&U_col_nnz, sizeof(int64_t) * A_cols));

    /* First, find number of nonzeros in the rows of M=(L+U) (with fill) */
    cuda_kernel_splu_symbolic_fact_trav_nnz<<<num_blocks_symb, num_threads_symb, 0, main_stream>>>(
        A_rows, A_cols, tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
        vert_fill, vert_queue, vert_mask, As_row_nnz);
    cuda_check_kernel_launch_err();

    /* From the row nnz, compute row pointers */
    cuda_check_err(cudaMemsetAsync(As_row_indptr_raw, 0, sizeof(int64_t), main_stream));
    cub_cumsum(As_row_nnz, As_row_indptr_raw + 1, A_rows, main_stream);
    cuda_check_err(cudaFreeAsync(As_row_nnz, main_stream));

    /* Allocate storage for the data and row indices arrays */
    int64_t As_nnz;
    cudaMemcpy(&As_nnz, &(As_row_indptr_raw[A_rows]), sizeof(int64_t), cudaMemcpyDeviceToHost);
    torch::Tensor As_indptr = torch::from_blob(As_row_indptr_raw, { static_cast<int64_t>(A_rows + 1) }, cudaFree, int_tens_opts);
    torch::Tensor As_indices = torch::empty({As_nnz}, int_tens_opts);
    torch::Tensor As_data = torch::empty({As_nnz}, scalar_tens_opts);

    /* Now, fill in As with row indices and entries of A (with explicit zeros where we are anticipating fill) */
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splu_cuda", ([&] {
        cuda_kernel_splu_symbolic_fact_trav_populate<<<num_blocks_symb, num_threads_symb, 0, main_stream>>>(
            A_rows, A_cols, tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t),
            vert_fill, vert_queue, vert_mask, tensor_acc(As_data, scalar_t), tensor_acc(As_indices, int64_t), As_row_indptr_raw);
    }));
    cuda_check_kernel_launch_err();
    cuda_check_err(cudaFreeAsync(vert_fill, main_stream));
    cuda_check_err(cudaFreeAsync(vert_queue, main_stream));
    cuda_check_err(cudaFreeAsync(vert_mask, main_stream));

    /* Compute the transpose/csc representation of As so that we have easy column access. */
    auto AsT = csr_transpose_forward_cuda(A_rows, A_cols, As_data, As_indices, As_indptr);
    torch::Tensor AsT_data = AsT[0];
    torch::Tensor AsT_indices = AsT[1];
    torch::Tensor AsT_indptr = AsT[2];

    /* Perform the numeric factorization on the CSC representation */
    cuda_kernel_count_U_nnz<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows, A_cols, tensor_acc(AsT_indices, int64_t), tensor_acc(AsT_indptr, int64_t), U_col_nnz);
    cuda_check_kernel_launch_err();

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splu_cuda", ([&] {
        cuda_kernel_splu_numeric_sflu<<<(A_cols + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols,
            tensor_acc(AsT_data, scalar_t), tensor_acc(AsT_indices, int64_t), tensor_acc(AsT_indptr, int64_t), U_col_nnz);
    }));
    cuda_check_kernel_launch_err();
    cuda_check_err(cudaFreeAsync(U_col_nnz, main_stream));

    /* Transpose back into CSR format */
    auto As_f = csr_transpose_forward_cuda(A_cols, A_rows, AsT_data, AsT_indices, AsT_indptr);
    return {As_f[0], As_f[1], As_f[2], AsT_data, AsT_indices, AsT_indptr};
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spsolve_backward,
               torch::Tensor grad_x, torch::Tensor x,
               int A_rows, int A_cols,
               torch::Tensor Mt_data, torch::Tensor Mt_indices, torch::Tensor Mt_indptr,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               torch::Tensor Pr, torch::Tensor Pc) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* grad_b = A^{-T} grad_x */
    torch::Tensor grad_b_y = sptrsv_forward_cuda(A_rows, A_cols, Mt_data, Mt_indices, Mt_indptr, true, false, permute_inverse_cuda(grad_x, Pc));
    torch::Tensor grad_b = permute_cuda(sptrsv_forward_cuda(A_rows, A_cols, Mt_data, Mt_indices, Mt_indptr, false, true, grad_b_y), Pr);

    /* grad_A = (-grad_b x^T) (*) mask(A) */
    torch::Tensor grad_A_data = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "sptrsv_backward_cuda", ([&] {
        cuda_kernel_masked_outerproduct<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows, A_cols, -1., tensor_acc(grad_b, scalar_t), tensor_acc(x, scalar_t),
            tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t), tensor_acc(grad_A_data, scalar_t));
    }));
    cuda_check_kernel_launch_err();

    return {grad_A_data, grad_b};
}

/* CSR To dense */
template <typename scalar_t>
__global__ void kernel_csr_to_dense_forward(
    int A_rows, int A_cols,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A_dense,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        const int64_t col = A_indices[row_i];
        A_dense[row][col] = A_data[row_i];
    }
}

FUNC_IMPL_CUDA(torch::Tensor,
              csr_to_dense_forward,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    torch::Tensor A_d = torch::zeros({A_rows, A_cols}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_to_dense_forward_cuda", ([&] {
        kernel_csr_to_dense_forward<<<
            (A_rows + threads_per_block - 1) / threads_per_block,
            threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc_3(A_d, 2, scalar_t),
                tensor_acc(A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return A_d;
}

template <typename scalar_t>
__global__ void kernel_csr_to_dense_backward(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> A_dense_grad,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data_grad,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        const int64_t col = A_indices[row_i];
        A_data_grad[row_i] = A_dense_grad[row][col];
    }
}

FUNC_IMPL_CUDA(torch::Tensor,
              csr_to_dense_backward,
              torch::Tensor grad_Ad,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A_data = torch::empty_like(A_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_to_dense_backward_cuda", ([&] {
        kernel_csr_to_dense_backward<<<
            (A_rows + threads_per_block - 1) / threads_per_block,
            threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc_3(grad_Ad, 2, scalar_t),
                tensor_acc(grad_A_data, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return grad_A_data;
}

/* CSR Row sum */
template <typename scalar_t>
__global__ void kernel_csr_row_sum_forward(
    int A_rows, int A_cols,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    scalar_t acc = 0.;
    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        acc += A_data[row_i];
    }
    x[row] = acc;
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_row_sum_forward,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    torch::Tensor x = torch::empty({A_rows}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_row_sum_forward_cuda", ([&] {
        kernel_csr_row_sum_forward<<<
            (A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(x, scalar_t), tensor_acc(A_data, scalar_t),
                tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return x;
}

template <typename scalar_t>
__global__ void kernel_csr_row_sum_backward(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    const scalar_t grad_row = grad_x[row];
    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        grad_A_data[row_i] = grad_row;
    }
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_row_sum_backward,
               torch::Tensor grad_x,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A_data = torch::empty_like(A_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_row_sum_backward_cuda", ([&] {
        kernel_csr_row_sum_backward<<<
            (A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(grad_x, scalar_t), tensor_acc(grad_A_data, scalar_t),
                tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return grad_A_data;
}


/* CSR Extract diagonal */
template <typename scalar_t>
__global__ void kernel_csr_extract_diagonal_forward(
    int A_rows, int A_cols,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    scalar_t diag_val = 0.;
    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        const int64_t col = A_indices[row_i];
        if (row == col) {
            diag_val = A_data[row_i];
        }
    }
    x[row] = diag_val;
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_extract_diagonal_forward,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    torch::Tensor x = torch::empty({A_rows}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_extract_diagonal_forward_cuda", ([&] {
        kernel_csr_extract_diagonal_forward<<<
            (A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(x, scalar_t), tensor_acc(A_data, scalar_t),
                tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return x;
}

template <typename scalar_t>
__global__ void kernel_csr_extract_diagonal_backward(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_x,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> grad_A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr) {

    const int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) {
        return;
    }

    const scalar_t grad_row = grad_x[row];
    for (int64_t row_i = A_indptr[row]; row_i < A_indptr[row + 1]; row_i++) {
        const int64_t col = A_indices[row_i];
        const scalar_t val = (row == col) ? grad_row : 0.;
        grad_A_data[row_i] = val;
    }
}

FUNC_IMPL_CUDA(torch::Tensor,
               csr_extract_diagonal_backward,
               torch::Tensor grad_x,
               int A_rows, int A_cols,
               torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A_data = torch::empty_like(A_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_extract_diagonal_backward_cuda", ([&] {
        kernel_csr_extract_diagonal_backward<<<
            (A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                A_rows, A_cols, tensor_acc(grad_x, scalar_t), tensor_acc(grad_A_data, scalar_t),
                tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));

    return grad_A_data;
}

FUNC_IMPL_CUDA(torch::Tensor,
               permute,
               torch::Tensor x, torch::Tensor P) {
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor xp = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "permute_cuda", ([&] {
        cuda_kernel_tensor_permute<<<
            (x.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                x.size(0), tensor_acc(x, scalar_t), tensor_acc(xp, scalar_t), tensor_acc(P, int64_t), true);
    }));

    return xp;
}

FUNC_IMPL_CUDA(torch::Tensor,
               permute_inverse,
               torch::Tensor x, torch::Tensor P) {
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor xp = torch::empty_like(x);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "permute_inverse_cuda", ([&] {
        cuda_kernel_tensor_permute<<<
            (x.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
                x.size(0), tensor_acc(x, scalar_t), tensor_acc(xp, scalar_t), tensor_acc(P, int64_t), false);
    }));

    return xp;
}
