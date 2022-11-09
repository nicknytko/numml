#include <torch/extension.h>
#include <map>
#include <tuple>

#include "sparse_csr.hpp"

/* Sparse GEMV */

FUNC_IMPL_CPU(torch::Tensor,
              spgemv_forward,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
              torch::Tensor x) {
    auto options = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    torch::Tensor Ax = torch::empty({A_rows}, options);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_col_ind.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_rowptr.accessor<int64_t, 1>();
        const auto x_acc = x.accessor<scalar_t, 1>();
        auto Ax_acc = Ax.accessor<scalar_t, 1>();

        for (int64_t row_i = 0; row_i < A_rows; row_i++) {
            scalar_t aij = 0.;
            for (int64_t col_j = A_indptr_acc[row_i]; col_j < A_indptr_acc[row_i + 1]; col_j++) {
                aij += A_data_acc[col_j] * x_acc[A_indices_acc[col_j]];
            }
            Ax_acc[row_i] = aij;
        }
    }));

    return Ax;
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              spgemv_backward,
              torch::Tensor grad_z, int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
              torch::Tensor x) {

    torch::Tensor grad_A = torch::empty_like(A_data);
    torch::Tensor grad_x = torch::zeros_like(x);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemv_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_col_ind.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_rowptr.accessor<int64_t, 1>();
        const auto x_acc = x.accessor<scalar_t, 1>();
        const auto grad_z_acc = grad_z.accessor<scalar_t, 1>();
        auto grad_A_acc = grad_A.accessor<scalar_t, 1>();
        auto grad_x_acc = grad_x.accessor<scalar_t, 1>();

        /* grad_A = alpha * outer(grad_w, x) (*) mask(A) */
        for (int64_t row = 0; row < A_rows; row++) {
            for (int64_t i = A_indptr_acc[row]; i < A_indptr_acc[row + 1]; i++) {
                const int64_t col = A_indices_acc[i];
                grad_A_acc[i] = grad_z_acc[row] * x_acc[col];
            }
        }

        /* grad_x = alpha * A^T grad_z */
        for (int64_t row = 0; row < A_rows; row++) {
            for (int64_t i = A_indptr_acc[row]; i < A_indptr_acc[row + 1]; i++) {
                const int64_t col = A_indices_acc[i];
                grad_x_acc[col] += grad_z_acc[row] * A_data_acc[i];
            }
        }
    }));

    return {grad_A, grad_x};
}

/* Sparse GEMM */

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              spgemm_forward,
              int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    int C_rows = A_rows;
    int C_cols = B_cols;

    /* Start with a sparse dict representation of C */
    std::vector<std::map<int, float>> C_dict;
    C_dict.resize(C_rows);
    int64_t C_nnz = 0;

    /* Do the matrix product by partial sums over each entry of A and rows of B */
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();

        const auto B_data_acc = B_data.accessor<scalar_t, 1>();
        const auto B_indices_acc = B_indices.accessor<int64_t, 1>();
        const auto B_indptr_acc = B_indptr.accessor<int64_t, 1>();

        for (int64_t A_row = 0; A_row < A_rows; A_row++) {
            for (int64_t A_i = A_indptr_acc[A_row]; A_i < A_indptr_acc[A_row + 1]; A_i++) {
                const int64_t A_col = A_indices_acc[A_i];

                for (int64_t B_i = B_indptr_acc[A_col]; B_i < B_indptr_acc[A_col + 1]; B_i++) {
                    const int64_t B_col = B_indices_acc[B_i];

                    const int64_t i = A_row;
                    const int64_t j = B_col;

                    if (C_dict[i].find(j) == C_dict[i].end()) {
                        /* Create the entry if it doesn't exist */
                        C_dict[i][j] = 0.f;
                        C_nnz ++;
                    }

                    C_dict[i][j] = C_dict[i][j] + A_data_acc[A_i] * B_data_acc[B_i];
                }
            }
        }
    }));

    /* Convert to CSR representation */
    torch::Tensor C_data = torch::empty(C_nnz, A_data.dtype());
    torch::Tensor C_indices = torch::empty(C_nnz, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor C_indptr = torch::empty(C_rows + 1, torch::TensorOptions().dtype(torch::kLong));

    int C_i = 0;
    for (int C_row = 0; C_row < C_rows; C_row++) {
        C_indptr[C_row] = C_i;

        for (const auto& colval : C_dict[C_row]) {
            C_data[C_i] = colval.second;
            C_indices[C_i] = colval.first;
            C_i ++;
        }
    }
    C_indptr[C_rows] = C_nnz;

    return {C_data, C_indices, C_indptr};
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              spgemm_backward,
              torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
              int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    const int64_t C_rows = A_rows;
    const int64_t C_cols = B_cols;

    torch::Tensor grad_A = torch::empty_like(A_data);
    torch::Tensor grad_B = torch::zeros_like(B_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spgemm_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        auto grad_A_data_acc = grad_A.accessor<scalar_t, 1>();

        const auto B_data_acc = B_data.accessor<scalar_t, 1>();
        const auto B_indices_acc = B_indices.accessor<int64_t, 1>();
        const auto B_indptr_acc = B_indptr.accessor<int64_t, 1>();
        auto grad_B_data_acc = grad_B.accessor<scalar_t, 1>();

        const auto grad_C_data_acc = grad_C.accessor<scalar_t, 1>();
        const auto C_indices_acc = C_indices.accessor<int64_t, 1>();
        const auto C_indptr_acc = C_indptr.accessor<int64_t, 1>();

        /** dA = (grad_C * B^T) (*) mask(A) */

        /* First build a map from i,j coordinates to indices in the data */
        std::map<std::tuple<int64_t, int64_t>, scalar_t> A_mask;
        for (int64_t A_row = 0; A_row < A_rows; A_row++) {
            for (int64_t A_i = A_indptr_acc[A_row]; A_i < A_indptr_acc[A_row + 1]; A_i++) {
                A_mask[std::make_tuple(A_row, A_indices_acc[A_i])] = A_i;
            }
        }

        /* Now, compute grad_C * B^T */
        for (int64_t C_row = 0; C_row < C_rows; C_row++) {
            for (int64_t B_row = 0; B_row < B_rows; B_row++) {
                if (A_mask.find(std::make_tuple(C_row, B_row)) == A_mask.end()) {
                    continue; /* This entry doesn't exist in the map, skip */
                }

                int64_t C_row_i = C_indptr_acc[C_row];
                int64_t B_row_i = B_indptr_acc[B_row];

                const int64_t C_row_end = C_indptr_acc[C_row + 1];
                const int64_t B_row_end = B_indptr_acc[B_row + 1];

                scalar_t aij = 0.;

                while (C_row_i < C_row_end && B_row_i < B_row_end) {
                    const int64_t C_col = C_indices_acc[C_row_i];
                    const int64_t B_col = B_indices_acc[B_row_i];

                    if (C_col < B_col) {
                        C_row_i ++;
                    } else if (C_col > B_col) {
                        B_row_i ++;
                    } else {
                        aij += grad_C_data_acc[C_row_i] * B_data_acc[B_row_i];
                        C_row_i ++;
                        B_row_i ++;
                    }
                }

                grad_A_data_acc[A_mask[std::make_tuple(C_row, B_row)]] = aij;
            }
        }

        /** dB = (A^T * grad_C) (*) mask(B) */

        /* Build index map */
        std::map<std::tuple<int64_t, int64_t>, scalar_t> B_mask;
        for (int64_t B_row = 0; B_row < B_rows; B_row++) {
            for (int64_t B_i = B_indptr_acc[B_row]; B_i < B_indptr_acc[B_row + 1]; B_i++) {
                B_mask[std::make_tuple(B_row, B_indices_acc[B_i])] = B_i;
            }
        }

        /* Compute A^T * grad_C */
        for (int64_t A_row = 0; A_row < A_rows; A_row++) {
            for (int64_t A_row_i = A_indptr_acc[A_row]; A_row_i < A_indptr_acc[A_row + 1]; A_row_i++) {
                const int64_t A_col = A_indices_acc[A_row_i];
                for (int64_t C_row_i = C_indptr_acc[A_row]; C_row_i < C_indptr_acc[A_row + 1]; C_row_i++) {
                    const int64_t C_col = C_indices_acc[C_row_i];

                    if (B_mask.find(std::make_tuple(A_col, C_col)) != B_mask.end()) {
                        /* Only look at indices in the mask */
                        grad_B_data_acc[B_mask[std::make_tuple(A_col,C_col)]] += A_data_acc[A_row_i] * grad_C_data_acc[C_row_i];
                    }
                }
            }
        }
    }));

    return {grad_A, grad_B};
}

/* CSR Transpose */

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              csr_transpose_forward,
              int A_rows, int A_columns,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    /* Based on the implementation from Scipy:
       https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L380 */

    const int64_t nnz = A_indptr[A_rows].item<int64_t>();

    torch::Tensor At_data = torch::empty(nnz);
    torch::Tensor At_indptr = torch::zeros(A_columns + 1, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor At_indices = torch::empty(nnz, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor At_to_A_idx = torch::empty(nnz, torch::TensorOptions().dtype(torch::kLong));

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "csr_transpose_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        auto At_data_acc = At_data.accessor<scalar_t, 1>();
        auto At_indices_acc = At_indices.accessor<int64_t, 1>();
        auto At_indptr_acc = At_indptr.accessor<int64_t, 1>();
        auto At_to_A_idx_acc = At_to_A_idx.accessor<int64_t, 1>();

        /* Compute number of nonzeros per column of A */
        for (int64_t i = 0; i < nnz; i++) {
            At_indptr[A_indices[i]] += 1;
        }

        /* Now, compute the cumulative sum of nnz to get starting rowptrs of A^T */
        int64_t cumsum = 0;
        for (int64_t column = 0; column < A_columns; column++) {
            const int64_t old_idx = At_indptr_acc[column];
            At_indptr_acc[column] = cumsum;
            cumsum += old_idx;
        }
        At_indptr_acc[A_columns] = nnz;

        /* Move data values into their correct spots */
        torch::Tensor At_row_acc = At_indptr.clone();
        auto At_row_acc_acc = At_row_acc.accessor<int64_t, 1>();

        for (int64_t row = 0; row < A_rows; row ++) {
            for (int64_t i = A_indptr_acc[row]; i < A_indptr_acc[row + 1]; i++) {
                const int64_t column = A_indices_acc[i];
                const int64_t dest = At_row_acc_acc[column];

                At_indices_acc[dest] = row;
                At_data_acc[dest] = A_data_acc[i];
                At_to_A_idx_acc[dest] = i;

                At_row_acc_acc[column] += 1;
            }
        }
    }));

    return {At_data, At_indices, At_indptr, At_to_A_idx};
}

FUNC_IMPL_CPU(torch::Tensor,
              csr_transpose_backward,
              torch::Tensor grad_At, torch::Tensor At_to_A_idx) {
    torch::Tensor grad_A = torch::empty_like(grad_At);

    AT_DISPATCH_FLOATING_TYPES(grad_At.type(), "csr_transpose_backward_cpu", ([&] {
        const auto grad_At_acc = grad_At.accessor<scalar_t, 1>();
        const auto At_to_A_idx_acc = At_to_A_idx.accessor<int64_t, 1>();
        auto grad_A_acc = grad_A.accessor<scalar_t, 1>();

        for (int i = 0; i < grad_At.size(0); i++) {
            grad_A_acc[At_to_A_idx_acc[i]] = grad_At_acc[i];
        }
    }));

    return grad_A;
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              splincomb_forward,
              int rows, int cols,
              torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64);

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype());

    torch::Tensor C_data_tens;
    torch::Tensor C_indices_tens;
    torch::Tensor C_indptr_tens;

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splincomb_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();

        const auto B_data_acc = B_data.accessor<scalar_t, 1>();
        const auto B_indices_acc = B_indices.accessor<int64_t, 1>();
        const auto B_indptr_acc = B_indptr.accessor<int64_t, 1>();

        std::vector<scalar_t> C_data;
        std::vector<int64_t> C_indices;
        std::vector<int64_t> C_indptr;

        const scalar_t alpha_c = alpha.item<scalar_t>();
        const scalar_t beta_c = beta.item<scalar_t>();

        /* Reserve a conservative guess for the nonzeros */
        C_data.reserve(std::max(A_data.size(0), B_data.size(0)));
        C_indices.reserve(std::max(A_data.size(0), B_data.size(0)));
        C_indptr.reserve(rows + 1);

        for (int64_t row = 0; row < rows; row++) {
            /* Indptr for this row is where we are in data array. */
            C_indptr.push_back(C_data.size());

            int64_t i_A = A_indptr_acc[row];
            int64_t i_B = B_indptr_acc[row];

            const int64_t end_A = A_indptr_acc[row + 1];
            const int64_t end_B = B_indptr_acc[row + 1];

            /* Merge the row of A and B */
            while (i_A < end_A && i_B < end_B) {
                const int64_t col_A = A_indices_acc[i_A];
                const int64_t col_B = B_indices_acc[i_B];

                if (col_A < col_B) {
                    C_data.push_back(alpha_c * A_data_acc[i_A]);
                    C_indices.push_back(col_A);
                    i_A++;
                } else if (col_A > col_B) {
                    C_data.push_back(beta_c * B_data_acc[i_B]);
                    C_indices.push_back(col_B);
                    i_B++;
                } else { /* we hit the same row-column pair in both matrices */
                    C_data.push_back(alpha_c * A_data_acc[i_A] + beta_c * B_data_acc[i_B]);
                    C_indices.push_back(col_A);
                    i_A++;
                    i_B++;
                }
            }

            /* Exhausted shared indices, now we add the rest of the row of A or B */
            while (i_A < end_A) {
                C_data.push_back(alpha_c * A_data_acc[i_A]);
                C_indices.push_back(A_indices_acc[i_A]);
                i_A++;
            }
            while (i_B < end_B) {
                C_data.push_back(beta_c * B_data_acc[i_B]);
                C_indices.push_back(B_indices_acc[i_B]);
                i_B++;
            }
        }

        C_indptr.push_back(C_data.size());

        C_data_tens = torch::from_blob(C_data.data(), {static_cast<int64_t>(C_data.size())}, scalar_tens_opts).clone();
        C_indices_tens = torch::from_blob(C_indices.data(), {static_cast<int64_t>(C_indices.size())}, int_tens_opts).clone();
        C_indptr_tens = torch::from_blob(C_indptr.data(), {static_cast<int64_t>(C_indptr.size())}, int_tens_opts).clone();
    }));

    return {
        C_data_tens,
        C_indices_tens,
        C_indptr_tens
    };
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              splincomb_backward,
              int rows, int cols,
              torch::Tensor alpha, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              torch::Tensor beta, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr,
              torch::Tensor grad_C_data, torch::Tensor C_indices, torch::Tensor C_indptr) {

    torch::Tensor grad_A = torch::empty_like(A_data);
    torch::Tensor grad_B = torch::empty_like(B_data);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splincomb_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        auto grad_A_acc = grad_A.accessor<scalar_t, 1>();

        const auto B_data_acc = B_data.accessor<scalar_t, 1>();
        const auto B_indices_acc = B_indices.accessor<int64_t, 1>();
        const auto B_indptr_acc = B_indptr.accessor<int64_t, 1>();
        auto grad_B_acc = grad_B.accessor<scalar_t, 1>();

        const auto C_indices_acc = C_indices.accessor<int64_t, 1>();
        const auto C_indptr_acc = C_indptr.accessor<int64_t, 1>();
        const auto grad_C_acc = grad_C_data.accessor<scalar_t, 1>();

        const scalar_t alpha_c = alpha.item<scalar_t>();
        const scalar_t beta_c = beta.item<scalar_t>();

        /* grad_A = alpha * grad_c (*) mask(A) */
        for (int64_t row = 0; row < rows; row++) {
            int64_t A_i = A_indptr_acc[row];
            int64_t C_i = C_indptr_acc[row];

            const int64_t A_end = A_indptr_acc[row + 1];
            const int64_t C_end = C_indptr_acc[row + 1];

            while (A_i < A_end && C_i < C_end) {
                const int64_t A_col = A_indices_acc[A_i];
                const int64_t C_col = C_indices_acc[C_i];

                if (A_col < C_col) {
                    A_i++;
                } else if (A_col > C_col) {
                    C_i++;
                } else {
                    grad_A_acc[A_i] = grad_C_acc[C_i] * alpha_c;
                    A_i++;
                    C_i++;
                }
            }
        }

        /* grad_B = beta * grad_c (*) mask(B) */
        for (int64_t row = 0; row < rows; row++) {
            int64_t B_i = B_indptr_acc[row];
            int64_t C_i = C_indptr_acc[row];

            const int64_t B_end = B_indptr_acc[row + 1];
            const int64_t C_end = C_indptr_acc[row + 1];

            while (B_i < B_end && C_i < C_end) {
                const int64_t B_col = B_indices_acc[B_i];
                const int64_t C_col = C_indices_acc[C_i];

                if (B_col < C_col) {
                    B_i++;
                } else if (B_col > C_col) {
                    C_i++;
                } else {
                    grad_B_acc[B_i] = grad_C_acc[C_i] * beta_c;
                    B_i++;
                    C_i++;
                }
            }
        }
    }));

    return {grad_A, grad_B};
}

FUNC_IMPL_CPU(torch::Tensor,
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
    torch::Tensor C = torch::empty({C_rows, C_cols}, scalar_tens_opts);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spdmm_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        const auto B_acc = B.accessor<scalar_t, 2>();
        auto C_acc = C.accessor<scalar_t, 2>();

        for (int64_t row = 0; row < C_rows; row++) {
            for (int64_t col = 0; col < C_cols; col++) {
                scalar_t cij = 0.;

                const int64_t next_row = A_indptr_acc[row + 1];
                for (int64_t row_i = A_indptr_acc[row]; row_i < next_row; row_i++) {
                    const int64_t k = A_indices_acc[row_i];
                    cij += A_data_acc[row_i] * B_acc[k][col];
                }

                C_acc[row][col] = cij;
            }
        }
    }));

    return C;
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              spdmm_backward,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              torch::Tensor B, torch::Tensor grad_C) {

    torch::Tensor grad_A = torch::empty_like(A_data);
    torch::Tensor grad_B = torch::zeros_like(B);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spdmm_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        const auto B_acc = B.accessor<scalar_t, 2>();
        const auto grad_C_acc = grad_C.accessor<scalar_t, 2>();
        auto grad_A_acc = grad_A.accessor<scalar_t, 1>();
        auto grad_B_acc = grad_B.accessor<scalar_t, 2>();

        const int64_t B_cols = B.size(1);

        /* grad_A = (grad_C * B^T) (*) mask(A) */
        for (int64_t row = 0; row < A_rows; row++) {
            for (int64_t row_i = A_indptr_acc[row]; row_i < A_indptr_acc[row + 1]; row_i++) {
                const int64_t col = A_indices_acc[row_i];
                scalar_t aij = 0.;

                for (int64_t k = 0; k < B_cols; k++) {
                    aij += grad_C_acc[row][k] * B_acc[col][k];
                }

                grad_A_acc[row_i] = aij;
            }
        }

        /* grad_B = (A^T grad_C) */
        for (int64_t k = 0; k < A_rows; k++) {
            for (int64_t i_i = A_indptr_acc[k]; i_i < A_indptr_acc[k + 1]; i_i++) {
                const int64_t i = A_indices_acc[i_i];
                for (int64_t j = 0; j < B_cols; j++) {
                    grad_B_acc[i][j] += A_data_acc[i_i] * grad_C_acc[k][j];
                }
            }
        }
    }));

    return {grad_A, grad_B};
}

FUNC_IMPL_CPU(torch::Tensor,
              sptrsv_forward,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              bool lower, bool unit, torch::Tensor b) {

    /* Compute in higher precision because single precision leads to catastrophic round-off errors. */
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(A_data.device().type(), A_data.device().index());
    torch::Tensor x_dbl = torch::empty({A_rows}, options);

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "sptrsv_forward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        const auto b_acc = b.accessor<scalar_t, 1>();
        auto x_acc = x_dbl.accessor<double, 1>();

        if (lower) {
            for (int64_t i = 0; i < A_rows; ++i) {
                double diag = 0.;
                double acc = 0.0;

                for (int64_t i_i = A_indptr_acc[i]; i_i < A_indptr_acc[i + 1]; ++i_i) {
                    const int64_t j = A_indices_acc[i_i];
                    const double Aij = static_cast<double>(A_data_acc[i_i]);

                    if (j > i) {
                        break;
                    } else if (j == i) {
                        diag = Aij;
                    } else {
                        acc += Aij * x_acc[j];
                    }
                }
                double xi = static_cast<double>(b_acc[i]) - acc;
                if (!unit) {
                    xi /= diag;
                }

                x_acc[i] = xi;
            }
        } else {
            for (int64_t i = A_rows - 1; i >= 0; --i) {
                double diag = 0.;
                double acc = 0.0;

                for (int64_t i_i = A_indptr_acc[i + 1] - 1; i_i >= A_indptr_acc[i]; --i_i) {
                    const int64_t j = A_indices_acc[i_i];
                    const double Aij = static_cast<double>(A_data_acc[i_i]);

                    if (j < i) {
                        break;
                    } else if (j == i) {
                        diag = Aij;
                    } else {
                        acc += Aij * x_acc[j];
                    }
                }
                double xi = static_cast<double>(b_acc[i]) - acc;
                if (!unit) {
                    xi /= diag;
                }

                x_acc[i] = xi;
            }
        }
    }));

    return x_dbl.to(A_data.dtype());
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              sptrsv_backward,
              torch::Tensor grad_x, torch::Tensor x,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
              bool lower, bool unit, torch::Tensor b) {

    /* Compute grad_b = A^{-T} grad_c */
    auto At = csr_transpose_forward_cpu(A_rows, A_cols, A_data, A_indices, A_indptr);
    torch::Tensor At_data = At[0];
    torch::Tensor At_indices = At[1];
    torch::Tensor At_indptr = At[2];

    torch::Tensor grad_b = sptrsv_forward_cpu(A_rows, A_cols, At_data, At_indices, At_indptr, !lower, unit, grad_x);

    /* Compute grad_A = -grad_B x^T (*) mask(A) */
    torch::Tensor grad_A_data = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "sptrsv_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        const auto x_acc = x.accessor<scalar_t, 1>();
        const auto grad_b_acc = grad_b.accessor<scalar_t, 1>();
        auto grad_A_data_acc = grad_A_data.accessor<scalar_t, 1>();

        for (int64_t i = 0; i < A_rows; i++) {
            for (int64_t i_i = A_indptr_acc[i]; i_i < A_indptr_acc[i+1]; i_i++) {
                const int64_t j = A_indices_acc[i_i];
                grad_A_data_acc[i_i] = -(grad_b_acc[i] * x_acc[j]);
            }
        }
    }));

    return {grad_A_data, grad_b};
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              splu,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    throw std::runtime_error("Sparse LU is not implemented on CPU.");
    return {};
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              spsolve_backward,
              torch::Tensor grad_x, torch::Tensor x,
              int A_rows, int A_cols,
              torch::Tensor Mt_data, torch::Tensor Mt_indices, torch::Tensor Mt_indptr,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    /* grad_b = A^{-T} grad_x */
    torch::Tensor grad_b_y = sptrsv_forward_cpu(A_rows, A_cols, Mt_data, Mt_indices, Mt_indptr, true, false, grad_x);
    torch::Tensor grad_b = sptrsv_forward_cpu(A_rows, A_cols, Mt_data, Mt_indices, Mt_indptr, false, true, grad_b_y);

    /* grad_A = (-grad_b x^T) (*) mask(A) */
    torch::Tensor grad_A_data = torch::empty_like(A_data);
    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "spsolve_backward_cpu", ([&] {
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();
        const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
        const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();
        const auto x_acc = x.accessor<scalar_t, 1>();
        const auto grad_b_acc = grad_b.accessor<scalar_t, 1>();
        auto grad_A_data_acc = grad_A_data.accessor<scalar_t, 1>();

        for (int64_t i = 0; i < A_rows; i++) {
            for (int64_t i_i = A_indptr_acc[i]; i_i < A_indptr_acc[i+1]; i_i++) {
                const int64_t j = A_indices_acc[i_i];
                grad_A_data_acc[i_i] = -(grad_b_acc[i] * x_acc[j]);
            }
        }
    }));

    return {grad_A_data, grad_b};
}
