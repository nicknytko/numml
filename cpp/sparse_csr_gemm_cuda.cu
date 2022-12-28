#include "cuda_common.cuh"
#include "sparse_csr.hpp"

#include <cuco/static_map.cuh>
#include <cuco/detail/hash_functions.cuh>
#include <cuco/allocator.hpp>

#include <chrono>

/**
 * Find the number of nonzeros for two sparse CSR matrices.
 * Indexed on rows of A.
 */
__global__ void cuda_kernel_find_nnz(
    int A_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_nnz) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= A_rows) {
        return;
    }
    A_nnz[i] = A_indptr[i+1] - A_indptr[i];
}

/**
 * Computes the maximal number of nonzero entries per row in the CSR SPGEMM product.
 * Indexed on rows of C.
 */
__global__ void cuda_kernel_find_Chat_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_nnz,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_nnz,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    int64_t nnz = 0;
    int64_t k;
    for (int64_t k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        nnz += B_nnz[k];
    }
    Chat_nnz[i] = nnz;
}


/**
 * Computes the matrix expansion in the CSR SPGEMM product.
 * Indexed on rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_Chat_expansion(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_J) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Grab starting index for entries of Chat */
    int64_t C_row_idx = 0;
    if (i > 0) {
        C_row_idx = Chat_indptr[i - 1];
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;
    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[k]; j_i < B_indptr[k + 1]; j_i++) {
            j = B_indices[j_i];

            Chat_I[C_row_idx] = i;
            Chat_J[C_row_idx] = j;
            Chat_data[C_row_idx] = A_data[k_i] * B_data[j_i];

            C_row_idx ++;
        }
    }
}

/**
 * Compute the number of nonzero entries per row of (sorted) Chat by counting unique column indices.
 * Indexed on rows of C.
 */
__global__ void cuda_kernel_Chat_to_C_row_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];

    int64_t cur_col = -1;
    int64_t nnz = 0;
    for (int64_t j = start_idx; j < end_idx; j++) {
        int64_t col = Chat_indices[j];
        if (col != cur_col) {
            nnz++;
        }
        cur_col = col;
    }
    C_nnz[i] = nnz;
}

/**
 * Assemble data and indices of C from Chat given indptr
 * Indexed on rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_assemble_C(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];
    int64_t cur_col = Chat_indices[start_idx];
    int64_t cur_C_ptr = C_indptr[i];
    scalar_t acc = 0.;

    for (int64_t j = start_idx; j <= end_idx; j++) {
        if (j == end_idx || Chat_indices[j] != cur_col) {
            C_data[cur_C_ptr] = acc;
            C_indices[cur_C_ptr] = cur_col;

            cur_C_ptr++;
            acc = 0.;

            if (j == end_idx) {
                break;
            } else {
                cur_col = Chat_indices[j];
            }
        }
        acc += Chat_data[j];
    }
}

/**
 * Assemble data and indices of C from Chat given indptr, does not affect the column indices.
 * Indexed on rows of A.
 */
template <typename scalar_t>
__global__ void cuda_kernel_assemble_C_data_only(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx = (i > 0 ? Chat_indptr[i-1] : 0);
    const int64_t end_idx = Chat_indptr[i];
    int64_t cur_col = Chat_indices[start_idx];
    int64_t cur_C_ptr = C_indptr[i];
    scalar_t acc = 0.;

    if (start_idx == end_idx) {
        C_data[cur_C_ptr] = 0.;
        return;
    }

    for (int64_t j = start_idx; j <= end_idx; j++) {
        if (j == end_idx || Chat_indices[j] != cur_col) {
            C_data[cur_C_ptr] = acc;

            cur_C_ptr++;
            acc = 0.;

            if (j == end_idx) {
                break;
            } else {
                cur_col = Chat_indices[j];
            }
        }
        acc += Chat_data[j];
    }
}

/* Sparse GEMM */

/**
 * CUDA implementation of sparse csr matrix matrix product (forward pass)
 * based on the algorithm sketched here:
 * http://lukeo.cs.illinois.edu/files/2015_BeDaOl_SPMM.pdf
 *
 * I don't claim that this is a super optimized version, but it's much faster than CPU :-)
 */

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemm_forward,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    const int C_rows = A_rows;
    const int C_cols = B_cols;

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Find NNZ in each matrix row */
    torch::Tensor A_nnz = torch::empty({A_rows}, int_tens_opts);
    torch::Tensor B_nnz = torch::empty({B_rows}, int_tens_opts);
    cuda_kernel_find_nnz<<<A_rows + threads_per_block - 1 / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows, tensor_acc(A_indptr, int64_t), tensor_acc(A_nnz, int64_t));
    cuda_check_kernel_launch_err();
    cuda_kernel_find_nnz<<<B_rows + threads_per_block - 1 / threads_per_block, threads_per_block, 0, main_stream>>>(
        B_rows, tensor_acc(B_indptr, int64_t), tensor_acc(B_nnz, int64_t));
    cuda_check_kernel_launch_err();

    /* Find NNZ in each row of \hat{C} */
    torch::Tensor Chat_nnz = torch::empty({C_rows}, int_tens_opts);
    cuda_kernel_find_Chat_nnz<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        C_rows, tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
        tensor_acc(A_nnz, int64_t), tensor_acc(B_nnz, int64_t), tensor_acc(Chat_nnz, int64_t));
    cuda_check_kernel_launch_err();

    torch::Tensor Chat_nnz_cumsum = Chat_nnz.cumsum(0);
    int64_t Chat_total_nnz = Chat_nnz_cumsum[Chat_nnz_cumsum.size(0) - 1].item<int64_t>();

    /* Compute the entries of Chat via expansion */
    torch::Tensor Chat_I = torch::empty({Chat_total_nnz}, int_tens_opts);
    torch::Tensor Chat_J = torch::empty({Chat_total_nnz}, int_tens_opts);
    torch::Tensor Chat_V = torch::empty({Chat_total_nnz}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_forward_cuda", ([&] {
        cuda_kernel_Chat_expansion<scalar_t><<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
            tensor_acc(B_data, scalar_t), tensor_acc(B_indptr, int64_t), tensor_acc(B_indices, int64_t),
            tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_V, scalar_t), tensor_acc(Chat_I, int64_t), tensor_acc(Chat_J, int64_t));
    }));
    cuda_check_kernel_launch_err();

    /* Lexicographically sort entries of Chat first by column index then by row index */
    lexsort_coo_ijv(Chat_I, Chat_J, Chat_V);

    /* Compute nonzeros in C by counting unique column indices in Chat */
    torch::Tensor C_nnz = torch::empty_like(Chat_nnz);
    cuda_kernel_Chat_to_C_row_nnz<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        C_rows, tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_J, int64_t), tensor_acc(C_nnz, int64_t));
    cuda_check_kernel_launch_err();

    /* Get C indptr by cumulative sum */
    torch::Tensor C_indptr_t = C_nnz.cumsum(0);
    torch::Tensor C_indptr = torch::zeros({C_rows + 1}, int_tens_opts);

    C_indptr.index_put_({torch::indexing::Slice(1, torch::indexing::None, torch::indexing::None)}, C_indptr_t);

    /* Now, assemble the matrix */
    int64_t C_total_nnz = C_indptr[C_indptr.size(0) - 1].item<int64_t>();
    torch::Tensor C_data = torch::empty({C_total_nnz}, scalar_tens_opts);
    torch::Tensor C_indices = torch::empty({C_total_nnz}, int_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_forward_cuda", [&] {
        cuda_kernel_assemble_C<<<(C_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows,
            tensor_acc(Chat_V, scalar_t), tensor_acc(Chat_nnz_cumsum, int64_t), tensor_acc(Chat_J, int64_t),
            tensor_acc(C_data, scalar_t), tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t));
    });
    cuda_check_kernel_launch_err();

    return {C_data, C_indices, C_indptr};
}

/**
 * Computes C = AB^T (*) mask(C)
 *
 * Assumes the CSR structure of C is given and nonzeros are to be computed.
 * Indexed on nonzeros of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_spmatmat_ABt_masked(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    int B_rows, int B_cols,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    int C_rows, int C_cols,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_data.size(0)) {
        return;
    }

    /* Find row and column index */
    int64_t col = C_indices[i];
    int64_t row = 0, row_l = 0, row_h = C_rows - 1;
    while (true) {
        int64_t row_m = (row_l + row_h)/2;

        if (i >= C_indptr[row_m] && i < C_indptr[row_m+1]) {
            row = row_m;
            break;
        } else if (i < C_indptr[row_m]) {
            row_h = row_m - 1;
        } else {
            row_l = row_m + 1;
        }
    }

    /* Compute this entry as dot product of rows of A and B */
    scalar_t cij = 0.;

    int A_row_i = A_indptr[row];
    int B_row_i = B_indptr[col];

    int A_row_end = A_indptr[row + 1];
    int B_row_end = B_indptr[col + 1];

    while (A_row_i < A_row_end && B_row_i < B_row_end) {
        int A_col = A_indices[A_row_i];
        int B_col = B_indices[B_row_i];

        if (A_col < B_col) {
            A_row_i ++;
        } else if (A_col > B_col) {
            B_row_i ++;
        } else {
            cij += A_data[A_row_i] * B_data[B_row_i];
            A_row_i ++;
            B_row_i ++;
        }
    }

    C_data[i] = cij;
}

/**
 * Computes the nonzeros in each row of the matrix expansion
 * in the CSR SPGEMM product, \hat{C}=A^TB.
 * Indexed on rows of A.
 */
__global__ void cuda_kernel_AT_Chat_expansion_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_nnz) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Multiply row of A and row of B */
    int64_t k_i, j_i;
    int64_t nnz = 0;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        nnz += (B_indptr[i + 1] - B_indptr[i]);
    }

    Chat_nnz[i] = nnz;
}

/**
 * Custom coordinate pair type for when we map i,j indices -> index in nonzeros array.
 */
struct coordinate_pair_t {
    int64_t row;
    int64_t col;

    __host__ __device__ coordinate_pair_t(int64_t _row, int64_t _col): row(_row), col(_col) {}

    __device__ bool operator==(const coordinate_pair_t& other) const {
        return (row == other.row && col == other.col);
    }
};

/**
 * Create a mapping between nonzero coordinates and their index in the data array.
 * Indexed on rows of the matrix.
 */
template <typename MapView>
__global__ void cuda_kernel_create_index_map(
    int A_rows, int A_cols,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    MapView A_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= A_rows) {
        return;
    }

    for (int64_t j_i = A_indptr[i]; j_i < A_indptr[i + 1]; j_i++) {
        int64_t j = A_indices[j_i];
        A_index_map.insert({coordinate_pair_t(i, j), j_i});
    }
}

/**
 * Compute nonzeros in the output of A^T Chat.
 */
template <typename MapView>
__global__ void cuda_kernel_masked_AT_Chat_expansion_nnz(
    int C_rows,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_nnz,
    MapView output_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;
    int64_t nnz = 0;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[i]; j_i < B_indptr[i + 1]; j_i++) {
            j = B_indices[j_i];

            if (output_index_map.contains(coordinate_pair_t(k, j))) {
                nnz++;
            }
        }
    }

    Chat_nnz[i] = nnz;
}

template <typename scalar_t, typename MapView>
__global__ void cuda_kernel_masked_AT_Chat_expansion(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_J,
    MapView output_index_map) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Grab starting index for entries of Chat */
    int64_t C_row_idx = 0;
    if (i > 0) {
        C_row_idx = Chat_indptr[i - 1];
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[i]; j_i < B_indptr[i + 1]; j_i++) {
            j = B_indices[j_i];

            if (output_index_map.contains(coordinate_pair_t(k, j))) {
                /* Insert the entry only if it exists in the sparsity mask */

                Chat_I[C_row_idx] = k;
                Chat_J[C_row_idx] = j;
                Chat_data[C_row_idx] = A_data[k_i] * B_data[j_i];

                C_row_idx ++;
            }
        }
    }
}

/**
 * Computes the matrix expansion in the CSR SPGEMM product, \hat{C}=A^TB (*) mask.
 * Indexed on rows of A.
 */
template <typename scalar_t>
__global__ void cuda_kernel_AT_Chat_expansion(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> A_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> A_indices,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> B_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> B_indices,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_J) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    /* Grab starting index for entries of Chat */
    int64_t C_row_idx = 0;
    if (i > 0) {
        C_row_idx = Chat_indptr[i - 1];
    }

    /* Multiply row of A and row of B */
    int64_t k_i, k, j_i, j;

    for (k_i = A_indptr[i]; k_i < A_indptr[i + 1]; k_i++) {
        k = A_indices[k_i];
        for (j_i = B_indptr[i]; j_i < B_indptr[i + 1]; j_i++) {
            j = B_indices[j_i];

            Chat_I[C_row_idx] = k;
            Chat_J[C_row_idx] = j;
            Chat_data[C_row_idx] = A_data[k_i] * B_data[j_i];

            C_row_idx ++;
        }
    }
}

__global__ void cuda_kernel_compute_indptr(
    int Chat_rows, const int tile_size,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_I) {

    const int64_t Chat_size = Chat_I.size(0);
    const int64_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t start_idx = thread_idx * tile_size;
    const int64_t end_idx = min((thread_idx + 1) * tile_size, Chat_size);

    if (start_idx >= end_idx) {
        return;
    }

    __syncthreads();

    int64_t cur_row = Chat_I[start_idx];
    for (int64_t i = start_idx; i < end_idx; i++) {
        if (Chat_I[i] != cur_row) {
            /* If we have hit an index w/ new row, then update all intermediate indptrs to
               start at the new index */
            const int64_t new_row = Chat_I[i];
            for (int64_t j = cur_row + 1; j <= new_row; j++) {
                Chat_indptr[j] = i;
            }
            cur_row = new_row;
        }
    }

    __syncthreads();

    if (end_idx < Chat_size &&
        Chat_I[end_idx] != cur_row) {
        const int64_t new_row = Chat_I[end_idx];
        for (int64_t j = cur_row + 1; j <= new_row; j++) {
            Chat_indptr[j] = end_idx;
        }
    } else if (end_idx == Chat_size) {
        for (int64_t j = cur_row + 1; j <= Chat_rows; j++) {
            Chat_indptr[Chat_rows] = Chat_size;
        }
    }
}

/**
 * Masked assembly from Chat to C, indexed on the rows of C.
 */
template <typename scalar_t>
__global__ void cuda_kernel_assemble_C_masked(
    int C_rows,
    const torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> Chat_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> Chat_indices,
    torch::PackedTensorAccessor64<scalar_t, 1, torch::RestrictPtrTraits> C_data,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indptr,
    const torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> C_indices) {

    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C_rows) {
        return;
    }

    const int64_t start_idx_Chat = Chat_indptr[i];
    const int64_t end_idx_Chat = Chat_indptr[i + 1];
    const int64_t start_idx_C = C_indptr[i];
    const int64_t end_idx_C = C_indptr[i + 1];
    int64_t cur_idx_Chat = start_idx_Chat;
    int64_t cur_col_Chat;
    int64_t cur_idx_C = start_idx_C;
    int64_t cur_col_C;
    scalar_t acc = 0.;

    while (cur_idx_Chat < end_idx_Chat && cur_idx_C < end_idx_C) {
        cur_col_Chat = Chat_indices[cur_idx_Chat];
        cur_col_C = C_indices[cur_idx_C];

        if (cur_col_Chat < cur_col_C) { /* Entry does not exist in C, skip... */
            cur_idx_Chat++;
        } else if (cur_col_Chat > cur_col_C) { /* No entry exists in Chat, zero this entry in C */
            cur_idx_C++;
        } else { /* Same column in Chat and C */
            acc += Chat_data[cur_idx_Chat];

            if (((cur_idx_Chat + 1) == end_idx_Chat) ||
                (Chat_indices[cur_idx_Chat + 1] != cur_col_Chat)) {
                /* If we are about to finish coalescing this column of Chat */
                C_data[cur_idx_C] = acc;
                acc = 0.;

                cur_idx_C++;
            }
            cur_idx_Chat++;
        }
    }
}

static torch::Tensor spgemm_backward_grad_B_square(torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    const int C_rows = A_rows;
    const int C_cols = B_cols;

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    /* Compute grad_B = (A^T grad_C) (*) mask(B) */
    const int64_t grad_B_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();

    /* Create mapping of grad_B nonzeros to indices in grad_B data array */
    const float map_load_factor = 0.6f;
    cuco::static_map<coordinate_pair_t, int64_t> grad_B_idx_map(
        static_cast<size_t>(static_cast<float>(grad_B_nnz) / map_load_factor),
        cuco::sentinel::empty_key(coordinate_pair_t(-1, -1)),
        cuco::sentinel::empty_value(int64_t(-1)),
        cuco::static_map<coordinate_pair_t, int64_t>::allocator_type{},
        main_stream);

    cuda_kernel_create_index_map<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        B_rows, B_cols, tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
        grad_B_idx_map.get_device_mutable_view());
    cuda_check_kernel_launch_err();

    /* Find NNZ in each row of \hat{grad_B}.  Note that this isn't actually a *row* in the
       output, but rather just a set of work for each thread to do. */
    torch::Tensor Bhat_nnz = torch::empty({A_rows}, int_tens_opts);
    cuda_kernel_masked_AT_Chat_expansion_nnz<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows,
        tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
        tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
        tensor_acc(Bhat_nnz, int64_t), grad_B_idx_map.get_device_view());
    cuda_check_kernel_launch_err();

    /* Cumulative sum to find starting point for each thread to write to. */
    torch::Tensor Bhat_nnz_cumsum = Bhat_nnz.cumsum(0);
    int64_t Bhat_total_nnz = Bhat_nnz_cumsum[Bhat_nnz_cumsum.size(0) - 1].item<int64_t>();

    /* Compute the entries of Bhat via masked expansion */
    torch::Tensor Bhat_I = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_J = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_V = torch::empty({Bhat_total_nnz}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(grad_C.type(), "spgemm_backward_cuda", ([&] {
        cuda_kernel_masked_AT_Chat_expansion<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
            tensor_acc(grad_C, scalar_t), tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
            tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_I, int64_t), tensor_acc(Bhat_J, int64_t),
            grad_B_idx_map.get_device_view());
    }));
    cuda_check_kernel_launch_err();

    /* Now, lexicographically sort entries of Bhat first by column index then by row index */
    lexsort_coo_ijv(Bhat_I, Bhat_J, Bhat_V);

    /* Find the ~actual~ number of nonzeros for each row of Bhat, now that we have the output */
    Bhat_nnz_cumsum = Bhat_I.bincount(c10::nullopt, B_rows).cumsum(0);

    /* Now, assemble the matrix */
    const int64_t B_total_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();
    torch::Tensor grad_B = torch::zeros({B_total_nnz}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cuda", [&] {
        cuda_kernel_assemble_C_data_only<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            B_rows,
            tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_J, int64_t),
            tensor_acc(grad_B, scalar_t), tensor_acc(B_indptr, int64_t));
    });
    cuda_check_kernel_launch_err();

    return grad_B;
}

static torch::Tensor spgemm_backward_grad_B_nonsquare(torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    const int C_rows = A_rows;
    const int C_cols = B_cols;

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    /* Compute grad_B = (A^T grad_C) (*) mask(B) */
    const int64_t grad_B_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();

    /* Find NNZ in each row of \hat{grad_B}.  Note that this isn't actually a *row* in the
       output, but rather just a set of work for each thread to do. */
    torch::Tensor Bhat_nnz = torch::empty({A_rows}, int_tens_opts);
    cuda_kernel_AT_Chat_expansion_nnz<<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
        A_rows,
        tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
        tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
        tensor_acc(Bhat_nnz, int64_t));
    cuda_check_kernel_launch_err();

    /* Cumulative sum to find starting point for each thread to write to. */
    torch::Tensor Bhat_nnz_cumsum = Bhat_nnz.cumsum(0);
    int64_t Bhat_total_nnz = Bhat_nnz_cumsum[Bhat_nnz_cumsum.size(0) - 1].item<int64_t>();

    /* Compute the entries of Bhat via masked expansion */
    torch::Tensor Bhat_I = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_J = torch::empty({Bhat_total_nnz}, int_tens_opts);
    torch::Tensor Bhat_V = torch::empty({Bhat_total_nnz}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(grad_C.type(), "spgemm_backward_cuda", ([&] {
        cuda_kernel_AT_Chat_expansion<scalar_t><<<(A_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            A_rows,
            tensor_acc(A_data, scalar_t), tensor_acc(A_indptr, int64_t), tensor_acc(A_indices, int64_t),
            tensor_acc(grad_C, scalar_t), tensor_acc(C_indptr, int64_t), tensor_acc(C_indices, int64_t),
            tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_I, int64_t), tensor_acc(Bhat_J, int64_t));
    }));
    cuda_check_kernel_launch_err();

    /* Now, lexicographically sort entries of Bhat first by column index then by row index */
    lexsort_coo_ijv(Bhat_I, Bhat_J, Bhat_V);

    /* Find the ~actual~ number of nonzeros for each row of Bhat, now that we have the output */
    Bhat_nnz_cumsum = torch::zeros({B_rows + 1}, int_tens_opts);
    const int tile_size = 64;
    const int num_threads = (Bhat_total_nnz + (threads_per_block * tile_size) - 1) /
        (threads_per_block * tile_size);
    cuda_kernel_compute_indptr<<<num_threads, threads_per_block, 0, main_stream>>>(
        B_rows, tile_size, tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_I, int64_t));
    cuda_check_kernel_launch_err();

    /* Now, assemble the matrix */
    const int64_t B_total_nnz = B_indptr[B_indptr.size(0) - 1].item<int64_t>();
    torch::Tensor grad_B = torch::zeros({B_total_nnz}, scalar_tens_opts);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cuda", [&] {
        cuda_kernel_assemble_C_masked<<<(B_rows + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            B_rows,
            tensor_acc(Bhat_V, scalar_t), tensor_acc(Bhat_nnz_cumsum, int64_t), tensor_acc(Bhat_J, int64_t),
            tensor_acc(grad_B, scalar_t), tensor_acc(B_indptr, int64_t), tensor_acc(B_indices, int64_t));
    });
    cuda_check_kernel_launch_err();

    return grad_B;
}

FUNC_IMPL_CUDA(std::vector<torch::Tensor>,
               spgemm_backward,
               torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
               int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
               int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {

    const int C_rows = A_rows;
    const int C_cols = B_cols;

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    /* Compute grad_A = (grad_C B^T) (*) mask(A) */
    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();
    torch::Tensor grad_A = torch::empty_like(A_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cuda", ([&] {
        cuda_kernel_spmatmat_ABt_masked<<<(A_data.size(0) + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            C_rows, C_cols, tensor_acc(grad_C, scalar_t), tensor_acc(C_indices, int64_t), tensor_acc(C_indptr, int64_t),
            B_rows, B_cols, tensor_acc(B_data, scalar_t), tensor_acc(B_indices, int64_t), tensor_acc(B_indptr, int64_t),
            A_rows, A_cols, tensor_acc(grad_A, scalar_t), tensor_acc(A_indices, int64_t), tensor_acc(A_indptr, int64_t));
    }));
    cuda_check_kernel_launch_err();

    torch::Tensor grad_B;
    if (false) {
        grad_B = spgemm_backward_grad_B_square(grad_C, C_indices, C_indptr,
                                               A_rows, A_cols, A_data, A_indices, A_indptr,
                                               B_rows, B_cols, B_data, B_indices, B_indptr);
    } else {
        grad_B = spgemm_backward_grad_B_nonsquare(grad_C, C_indices, C_indptr,
                                                  A_rows, A_cols, A_data, A_indices, A_indptr,
                                                  B_rows, B_cols, B_data, B_indices, B_indptr);
    }

    return {grad_A, grad_B};
}
