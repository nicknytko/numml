#include <torch/extension.h>
#include <map>
#include <tuple>

/* Sparse GEMV */

std::vector<torch::Tensor> spgemv_forward(int A_rows, int A_cols, torch::Tensor alpha,
                                          torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                                          torch::Tensor x, torch::Tensor beta, torch::Tensor y) {
    torch::Tensor Ax = torch::zeros({A_rows}, A_data.dtype());
    for (int row_i = 0; row_i < A_rows; row_i++) {
        for (int col_j = A_rowptr[row_i].item<int>(); col_j < A_rowptr[row_i + 1].item<int>(); col_j++) {
            Ax[row_i] += A_data[col_j] * x[A_col_ind[col_j]];
        }
    }
    return {Ax, alpha * Ax + beta * y};
}

std::vector<torch::Tensor> spgemv_backward(torch::Tensor grad_z, int A_rows, int A_cols, torch::Tensor alpha,
                                           torch::Tensor A_data, torch::Tensor A_col_ind, torch::Tensor A_rowptr,
                                           torch::Tensor x, torch::Tensor beta, torch::Tensor y) {

    /* grad_A = alpha * outer(grad_w, x) (*) mask(A) */
    torch::Tensor grad_A = torch::zeros_like(A_data);
    for (int row = 0; row < A_rows; row++) {
        for (int i = A_rowptr[row].item<int>(); i < A_rowptr[row + 1].item<int>(); i++) {
            int col = A_col_ind[i].item<int>();
            grad_A[i] = alpha * grad_z[row] * x[col];
        }
    }

    /* grad_x = alpha * A^T grad_z */
    torch::Tensor grad_x = torch::zeros_like(x);
    for (int row = 0; row < A_rows; row++) {
        for (int i = A_rowptr[row].item<int>(); i < A_rowptr[row + 1].item<int>(); i++) {
            int col = A_col_ind[i].item<int>();
            grad_x[col] += grad_z[row] * A_data[i] * alpha;
        }
    }

    return {grad_A, grad_x};
}

/* Sparse GEMM */

std::vector<torch::Tensor> spgemm_forward(int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                                          int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    int C_rows = A_rows;
    int C_cols = B_cols;

    /* Start with a sparse dict representation of C */
    std::vector<std::map<int, float>> C_dict;
    C_dict.resize(C_rows);

    int C_nnz = 0;

    /* Do the matrix product by partial sums over each entry of A and rows of B */
    for (int A_row = 0; A_row < A_rows; A_row++) {
        for (int A_i = A_indptr[A_row].item<int>(); A_i < A_indptr[A_row + 1].item<int>(); A_i++) {
            int A_col = A_indices[A_i].item<int>();

            for (int B_i = B_indptr[A_col].item<int>(); B_i < B_indptr[A_col + 1].item<int>(); B_i++) {
                int B_col = B_indices[B_i].item<int>();

                int i = A_row;
                int j = B_col;

                if (C_dict[i].find(j) == C_dict[i].end()) {
                    /* Create the entry if it doesn't exist */
                    C_dict[i][j] = 0.f;
                    C_nnz ++;
                }
                C_dict[i][j] = C_dict[i][j] + (A_data[A_i] * B_data[B_i]).item<float>();
            }
        }
    }

    /* Convert to CSR representation */
    torch::Tensor C_data = torch::zeros(C_nnz, A_data.dtype());
    torch::Tensor C_indices = torch::zeros(C_nnz, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor C_indptr = torch::zeros(C_rows + 1, torch::TensorOptions().dtype(torch::kLong));

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

std::vector<torch::Tensor> spgemm_backward(torch::Tensor grad_C, torch::Tensor C_indices, torch::Tensor C_indptr,
                                           int A_rows, int A_cols, torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr,
                                           int B_rows, int B_cols, torch::Tensor B_data, torch::Tensor B_indices, torch::Tensor B_indptr) {
    int C_rows = A_rows;
    int C_cols = B_cols;

    /** dA = (grad_C * B^T) (*) mask(A) */
    torch::Tensor grad_A = torch::zeros_like(A_data);

    /* First build a map from i,j coordinates to indices in the data */
    std::map<std::tuple<int, int>, float> A_mask;
    for (int A_row = 0; A_row < A_rows; A_row++) {
        for (int A_i = A_indptr[A_row].item<int>(); A_i < A_indptr[A_row + 1].item<int>(); A_i++) {
            A_mask[std::make_tuple(A_row, A_indices[A_i].item<int>())] = A_i;
        }
    }

    /* Now, compute grad_C * B^T */
    for (int C_row = 0; C_row < C_rows; C_row++) {
        for (int B_row = 0; B_row < B_rows; B_row++) {
            if (A_mask.find(std::make_tuple(C_row, B_row)) == A_mask.end()) {
                continue; /* This entry doesn't exist in the map, skip */
            }

            int C_row_i = (C_indptr[C_row]).item<int>();
            int B_row_i = (B_indptr[B_row]).item<int>();

            int C_row_end = (C_indptr[C_row + 1]).item<int>();
            int B_row_end = (B_indptr[B_row + 1]).item<int>();

            float aij = 0.f;

            while (C_row_i < C_row_end && B_row_i < B_row_end) {
                int C_col = C_indices[C_row_i].item<int>();
                int B_col = B_indices[B_row_i].item<int>();

                if (C_col < B_col) {
                    C_row_i ++;
                } else if (C_col > B_col) {
                    B_row_i ++;
                } else {
                    aij += (grad_C[C_row_i] * B_data[B_row_i]).item<float>();
                    C_row_i ++;
                    B_row_i ++;
                }
            }

            grad_A[A_mask[std::make_tuple(C_row, B_row)]] = aij;
        }
    }

    /** dB = (A^T * grad_C) (*) mask(B) */
    torch::Tensor grad_B = torch::zeros_like(B_data);

    /* Build index map */
    std::map<std::tuple<int, int>, float> B_mask;
    for (int B_row = 0; B_row < B_rows; B_row++) {
        for (int B_i = B_indptr[B_row].item<int>(); B_i < B_indptr[B_row + 1].item<int>(); B_i++) {
            B_mask[std::make_tuple(B_row, B_indices[B_i].item<int>())] = B_i;
        }
    }

    /* Compute A^T * grad_C */
    for (int A_row = 0; A_row < A_rows; A_row++) {
        for (int A_row_i = A_indptr[A_row].item<int>(); A_row_i < A_indptr[A_row + 1].item<int>(); A_row_i++) {
            int A_col = A_indices[A_row_i].item<int>();
            for (int C_row_i = C_indptr[A_row].item<int>(); C_row_i < C_indptr[A_row + 1].item<int>(); C_row_i++) {
                int C_col = C_indices[C_row_i].item<int>();

                if (B_mask.find(std::make_tuple(A_col, C_col)) != B_mask.end()) {
                    /* Only look at indices in the mask */
                    grad_B[B_mask[std::make_tuple(A_col,C_col)]] += (A_data[A_row_i] * grad_C[C_row_i]).item<float>();
                }
            }
        }
    }

    return {grad_A, grad_B};
}

/* CSR Transpose */

std::vector<torch::Tensor> csr_transpose_forward(int A_rows, int A_columns,
                                                 torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    /* Based on the implementation from Scipy:
       https://github.com/scipy/scipy/blob/3b36a574dc657d1ca116f6e230be694f3de31afc/scipy/sparse/sparsetools/csr.h#L380 */

    int nnz = A_indptr[A_rows].item<int>();

    torch::Tensor At_data = torch::zeros(nnz);
    torch::Tensor At_indptr = torch::zeros(A_columns + 1, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor At_indices = torch::zeros(nnz, torch::TensorOptions().dtype(torch::kLong));

    /* Compute number of nonzeros per column of A */
    for (int i = 0; i < nnz; i++) {
        At_indptr[A_indices[i]] += 1;
    }

    /* Now, compute the cumulative sum of nnz to get starting rowptrs of A^T */
    int cumsum = 0;
    for (int column = 0; column < A_columns; column++) {
        int old_idx = At_indptr[column].item<int>();
        At_indptr[column] = cumsum;
        cumsum += old_idx;
    }
    At_indptr[A_columns] = nnz;

    /* Move data values into their correct spots */
    torch::Tensor At_row_acc = At_indptr.clone();
    torch::Tensor At_to_A_idx = torch::zeros(nnz, torch::TensorOptions().dtype(torch::kLong));
    for (int row = 0; row < A_rows; row ++) {
        for (int i = A_indptr[row].item<int>(); i < A_indptr[row + 1].item<int>(); i++) {
            int column = A_indices[i].item<int>();
            int dest = At_row_acc[column].item<int>();

            At_indices[dest] = row;
            At_data[dest] = A_data[i];
            At_to_A_idx[dest] = i;

            At_row_acc[column] += 1;
        }
    }

    return {At_data, At_indices, At_indptr, At_to_A_idx};
}

torch::Tensor csr_transpose_backward(torch::Tensor grad_At, torch::Tensor At_to_A_idx) {
    torch::Tensor grad_A = torch::zeros_like(grad_At);

    for (int i = 0; i < grad_At.size(0); i++) {
        grad_A[At_to_A_idx[i].item<int>()] = grad_At[i];
    }

    return grad_A;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spgemv_forward", &spgemv_forward, "SPGEMV forward");
    m.def("spgemv_backward", &spgemv_backward, "SPGEMV backward");

    m.def("spgemm_forward", &spgemm_forward, "SPGEMM forward");
    m.def("spgemm_backward", &spgemm_backward, "SPGEMM backward");

    m.def("csr_transpose_forward", &csr_transpose_forward, "CSR Transpose forward");
    m.def("csr_transpose_backward", &csr_transpose_backward, "CSR Transpose backward");
}
