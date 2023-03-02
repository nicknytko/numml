#include <torch/extension.h>
#include <ATen/core/TensorAccessor.h>

#include "sparse_csr.hpp"
#include "vector_queue.hpp"
#include "superlu.hpp"

int64_t cpu_indices_binsearch(int64_t i_start, int64_t i_end, const int64_t i_search,
                              const torch::TensorAccessor<int64_t, 1, torch::DefaultPtrTraits>& indices) {
    int64_t i_mid;
    while (i_start <= i_end) {
        i_mid = (i_start + i_end) / 2;
        if (indices[i_mid] < i_search) {
            i_start = i_mid + 1;
        } else if (indices[i_mid] > i_search) {
            i_end = i_mid - 1;
        } else if (indices[i_mid] == i_search) {
            return i_mid;
        }
    }
    return -1;
}

FUNC_IMPL_CPU(std::vector<torch::Tensor>,
              splu,
              int A_rows, int A_cols,
              torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {

    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(A_data.device().type(), A_data.device().index());

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(A_data.dtype())
        .device(A_data.device().type(), A_data.device().index());

    const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
    const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();

    /* Metadata for symbolic fill2 algorithm */
    std::vector<int64_t> vert_fill(A_rows, 0);
    numml::vector_queue<int64_t> vert_queue(A_rows);
    std::vector<bool> vert_mask(A_cols, false);
    std::vector<int64_t> As_row_nnz(A_rows, 0);

    /* First, find number of nonzeros in the columns of M=(L+U) (with fill) */
    for (int64_t i = 0; i < A_rows; i++) {
        std::fill(vert_fill.begin(), vert_fill.end(), 0);
        std::fill(vert_mask.begin(), vert_mask.end(), false);
        /* Initialize the current node and all directly connected neighbors */
        vert_fill[i] = i;

        for (int64_t u_i = A_indptr_acc[i]; u_i < A_indptr_acc[i + 1]; u_i++) {
            const int64_t u = A_indices_acc[u_i];
            vert_fill[u] = i;
            vert_mask[u] = true;
        }

        /* Perform the traversal from fill2 */
        for (int64_t t = 0; t < i; t++) {
            if (vert_fill[t] != i) {
                continue;
            }

            vert_queue.clear();
            vert_queue.push_back(t);

            while (!vert_queue.empty()) {
                const int64_t u = vert_queue.pop_front();
                for (int64_t w_i = A_indptr_acc[u]; w_i < A_indptr_acc[u + 1]; w_i++) {
                    const int64_t w = A_indices_acc[w_i];
                    if (vert_fill[w] >= i) {
                        continue;
                    }

                    vert_fill[w] = i;
                    if (w > t) {
                        vert_mask[w] = true;
                    } else {
                        vert_queue.push_back(w);
                    }
                }
            }
        }

        /* Determine number of nonzeros in the row */
        int64_t nnz = 0;
        for (int64_t j = 0; j < A_cols; j++) {
            if (vert_mask[j]) {
                nnz++;
                vert_mask[j] = false;
            }
        }
        As_row_nnz[i] = nnz;
    }

    /* Compute row indptr as the cumulative sum over the row nnz */
    int64_t* As_row_indptr_raw = static_cast<int64_t*>(std::calloc(A_rows + 1, sizeof(int64_t)));
    for (int64_t i = 1; i < A_rows + 1; i++) {
        As_row_indptr_raw[i] = As_row_indptr_raw[i-1] + As_row_nnz[i-1];
    }

    /* Now, allocate the storage required and initialize As to be A with explicit zeros where fill
       is to be introduced. */
    const int64_t As_nnz = As_row_indptr_raw[A_rows];
    torch::Tensor As_indptr = torch::from_blob(As_row_indptr_raw, { static_cast<int64_t>(A_rows + 1) }, std::free, int_tens_opts);
    torch::Tensor As_indices = torch::empty({As_nnz}, int_tens_opts);
    torch::Tensor As_data = torch::empty({As_nnz}, scalar_tens_opts);

    /* Pre-define our return transposed types since we can't return from inside the lambda */
    torch::Tensor AsT_indptr, AsT_indices, AsT_data;

    AT_DISPATCH_FLOATING_TYPES(A_data.type(), "splu_cpu", ([&] {
        const auto As_indptr_acc = As_indptr.accessor<int64_t, 1>();
        auto As_indices_acc = As_indices.accessor<int64_t, 1>();
        auto As_data_acc = As_data.accessor<scalar_t, 1>();
        const auto A_data_acc = A_data.accessor<scalar_t, 1>();

        for (int64_t i = 0; i < A_rows; i++) {
            std::fill(vert_fill.begin(), vert_fill.end(), 0);
            std::fill(vert_mask.begin(), vert_mask.end(), false);
            /* Initialize the current node and all directly connected neighbors */
            vert_fill[i] = i;

            for (int64_t u_i = A_indptr_acc[i]; u_i < A_indptr_acc[i + 1]; u_i++) {
                const int64_t u = A_indices_acc[u_i];
                vert_fill[u] = i;
                vert_mask[u] = true;
            }

            /* Perform the traversal from fill2 */
            for (int64_t t = 0; t < i; t++) {
                if (vert_fill[t] != i) {
                    continue;
                }

                vert_queue.clear();
                vert_queue.push_back(t);

                while (!vert_queue.empty()) {
                    const int64_t u = vert_queue.pop_front();
                    for (int64_t w_i = A_indptr_acc[u]; w_i < A_indptr_acc[u + 1]; w_i++) {
                        const int64_t w = A_indices_acc[w_i];
                        if (vert_fill[w] >= i) {
                            continue;
                        }

                        vert_fill[w] = i;
                        if (w > t) {
                            vert_mask[w] = true;
                        } else {
                            vert_queue.push_back(w);
                        }
                    }
                }
            }

            /* Insert row indices and nonzero values of At_data.
               This is essentially a union of the two columns, where entries in As *only* are explicitly zero. */

            int64_t As_ptr = 0; /* Current entry in vert_visited array */
            int64_t A_ptr = A_indptr_acc[i]; /* Current index in original A */
            int64_t As_out_ptr = As_indptr_acc[i]; /* Current index in output As */

            const int64_t As_end = A_cols;
            const int64_t A_end = A_indptr_acc[i + 1];

            while (As_ptr < As_end && A_ptr < A_end) {
                /* Make sure we actually are at a nonzero of As */
                while (!vert_mask[As_ptr]) {
                    As_ptr++;
                }

                const int64_t As_col = As_ptr;
                const int64_t A_col = A_indices_acc[A_ptr];
                if (As_col < A_col) {
                    As_data_acc[As_out_ptr] = 0.;
                    As_indices_acc[As_out_ptr] = As_col;
                    vert_mask[As_ptr] = false;

                    As_ptr++;
                    As_out_ptr++;
                } else if (As_col > A_col) {
                    /* This is probably unlikely, since A is a subset of As..?
                       Nonetheless, let's add it here just in case. */
                    As_data_acc[As_out_ptr] = A_data_acc[A_ptr];
                    As_indices_acc[As_out_ptr] = A_col;

                    A_ptr++;
                    As_out_ptr++;
                } else { /* As_col == A_col */
                    As_data_acc[As_out_ptr] = A_data_acc[A_ptr];
                    As_indices_acc[As_out_ptr] = A_col;
                    vert_mask[As_ptr] = false;

                    A_ptr++;
                    As_ptr++;
                    As_out_ptr++;
                }
            }
            /* Finish off with rest of As entries */
            for (; As_ptr < As_end; As_ptr++) {
                if (vert_mask[As_ptr]) {
                    As_data_acc[As_out_ptr] = 0.;
                    As_indices_acc[As_out_ptr] = As_ptr;
                    As_out_ptr++;
                    vert_mask[As_ptr] = false;
                }
            }
        }

        /* Now that we have As, we can get As^T... */
        auto AsT = csr_transpose_forward_cpu(A_rows, A_cols, As_data, As_indices, As_indptr);
        AsT_data = AsT[0];
        AsT_indices = AsT[1];
        AsT_indptr = AsT[2];
        auto AsT_data_acc = AsT_data.accessor<scalar_t, 1>();
        const auto AsT_indptr_acc = AsT_indptr.accessor<int64_t, 1>();
        const auto AsT_indices_acc = AsT_indices.accessor<int64_t, 1>();

        /* ...and find the numeric factorization.  This is a serial version of the SFLU algorithm. */
        for (int64_t k = 0; k < A_cols; k++) {
            int64_t diag_idx;
            const int64_t col_end = AsT_indptr_acc[k + 1];
            for (int64_t i_i = AsT_indptr_acc[k]; i_i < col_end; i_i++) {
                const int64_t i = AsT_indices_acc[i_i];
                if (i == k) {
                    /* Stop once we get to the diagonal */
                    diag_idx = i_i;
                    break;
                }

                /* Left-looking product */
                for (int64_t j_i = i_i + 1; j_i < col_end; j_i++) {
                    const int64_t j = AsT_indices_acc[j_i];
                    const int64_t A_ji_i = cpu_indices_binsearch(AsT_indptr_acc[i], AsT_indptr_acc[i + 1] - 1, j, AsT_indices_acc);
                    if (A_ji_i == -1) {
                        continue;
                    }
                    const scalar_t A_ji = AsT_data_acc[A_ji_i];
                    const scalar_t A_ik = AsT_data_acc[i_i];

                    /* A_{jk} \gets A_{jk} - A_{ji} A_{ik} */
                    AsT_data[j_i] -= A_ji * A_ik;
                }
            }

            /* Divide column of L by diagonal entry of U */
            const scalar_t A_kk = AsT_data_acc[diag_idx];
            for (int64_t i = diag_idx + 1; i < AsT_indptr_acc[k + 1]; i++) {
                AsT_data_acc[i] /= A_kk;
            }
        }

        /* Transpose AsT back into As */
        auto As = csr_transpose_forward_cpu(A_cols, A_rows, AsT_data, AsT_indices, AsT_indptr);
        As_data = As[0];
        As_indices = As[1];
        As_indptr = As[2];
    }));

    return {As_data, As_indices, As_indptr, AsT_data, AsT_indices, AsT_indptr};
}
