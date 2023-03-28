#ifndef SUPERLU_HPP_
#define SUPERLU_HPP_

#include <vector>
#include <tuple>
#include <algorithm>

/* Include all the existing SuperLU headers in a namespace */
namespace superlu_c {
    /* Single precision */
    #include <slu_sdefs.h>

    /* Double precision */
    #include <slu_ddefs.h>

    /* Single-complex */
    #include <slu_cdefs.h>

    /* Double-complex */
    #include <slu_zdefs.h>

    /* Helper functions */
    #include <slu_util.h>
}

superlu_c::Dtype_t type_to_superlu_dtype(const std::type_info& type) {
    if (type == typeid(double)) {
        return superlu_c::SLU_D;
    } else if (type == typeid(float)) {
        return superlu_c::SLU_S;
    }
    throw std::runtime_error("Unknown type");
}

const c10::ScalarType type_to_torch_dtype(const std::type_info& type) {
    if (type == typeid(double)) {
        return torch::kFloat64;
    } else if (type == typeid(float)) {
        return torch::kFloat32;
    }
    throw std::runtime_error("Unknown type");
}

#define delete_if_exists(x) do {                \
    if ((x) != nullptr) {                       \
        std::free(x);                           \
        x = nullptr;                            \
    }} while (0);

template<typename T>
class SuperLUMatrix {
private:
    void copy_base_matrix(const superlu_c::SuperMatrix* mat) {
        std::memcpy(&matrix, mat, sizeof(superlu_c::SuperMatrix));

        if (mat->Store != nullptr) {
            if (mat->Stype == superlu_c::SLU_NC) {
                matrix.Store = new superlu_c::NCformat;
                superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(matrix.Store);
                superlu_c::NCformat* other_store = static_cast<superlu_c::NCformat*>(mat->Store);

                store->nnz = other_store->nnz;
                store->nzval = new T[store->nnz];
                store->rowind = new int32_t[store->nnz];
                store->colptr = new int32_t[matrix.ncol + 1];

                std::memcpy(store->nzval, other_store->nzval, sizeof(T) * store->nnz);
                std::memcpy(store->rowind, other_store->rowind, sizeof(int32_t) * store->nnz);
                std::memcpy(store->colptr, other_store->colptr, sizeof(int32_t) * matrix.ncol + 1);
            } else if (mat->Stype == superlu_c::SLU_SC) {
                matrix.Store = new superlu_c::SCformat;
                superlu_c::SCformat* store = static_cast<superlu_c::SCformat*>(matrix.Store);
                superlu_c::SCformat* other_store = static_cast<superlu_c::SCformat*>(mat->Store);

                store->nnz = other_store->nnz;
                store->nsuper = other_store->nsuper;
                store->nzval = new T[store->nnz];
                store->rowind = new int32_t[store->nnz];
                store->rowind_colptr = new int32_t[matrix.ncol + 1];
                store->col_to_sup = new int32_t[matrix.ncol];
                store->sup_to_col = new int32_t[store->nsuper + 2];

                std::memcpy(store->nzval, other_store->nzval, sizeof(T) * store->nnz);
                std::memcpy(store->rowind, other_store->rowind, sizeof(int32_t) * store->nnz);
                std::memcpy(store->rowind_colptr, other_store->rowind_colptr, sizeof(int32_t) * matrix.ncol + 1);
                std::memcpy(store->col_to_sup, other_store->col_to_sup, sizeof(int32_t) * matrix.ncol);
                std::memcpy(store->sup_to_col, other_store->sup_to_col, sizeof(int32_t) * store->nsuper + 2);
            } else {
                throw std::runtime_error("Unknown matrix type for copying.");
            }
        }
    }

    void free_self() {
        if (matrix.Store != nullptr) {
            if (matrix.Stype == superlu_c::SLU_NC) {
                superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(matrix.Store);

                delete_if_exists(store->nzval);
                delete_if_exists(store->rowind);
                delete_if_exists(store->colptr);
            } else if (matrix.Stype == superlu_c::SLU_SC) {
                superlu_c::SCformat* store = static_cast<superlu_c::SCformat*>(matrix.Store);

                delete_if_exists(store->nzval);
                delete_if_exists(store->nzval_colptr);
                delete_if_exists(store->rowind);
                delete_if_exists(store->rowind_colptr);
                delete_if_exists(store->col_to_sup);
                delete_if_exists(store->sup_to_col);
            }

            std::free(matrix.Store);
            matrix.Store = nullptr;
        }
    }

public:
    superlu_c::SuperMatrix matrix;

    SuperLUMatrix(bool allocate=true) {
        matrix.Stype = superlu_c::SLU_NC;
        matrix.Mtype = superlu_c::SLU_GE;
        matrix.Dtype = type_to_superlu_dtype(typeid(T));
        matrix.nrow = 0;
        matrix.ncol = 0;

        if (allocate) {
            matrix.Store = new superlu_c::NCformat;
            superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(matrix.Store);
            store->nnz = 0;
            store->nzval = nullptr;
            store->rowind = nullptr;
            store->colptr = nullptr;
        } else {
            matrix.Store = nullptr;
        }
    }

    SuperLUMatrix(const SuperLUMatrix& mat) {
        copy_base_matrix(&mat.matrix);
    }

    SuperLUMatrix(SuperLUMatrix&& mat) {
        std::memcpy(&matrix, &mat.matrix, sizeof(superlu_c::SuperMatrix));
        mat.matrix.Store = nullptr;
    }

    SuperLUMatrix(const superlu_c::SuperMatrix& mat) {
        copy_base_matrix(&mat);
    }

    SuperLUMatrix(superlu_c::SuperMatrix&& mat) {
        std::memcpy(&matrix, &mat, sizeof(superlu_c::SuperMatrix));
        mat.Store = nullptr;
    }

    SuperLUMatrix& operator=(const SuperLUMatrix& mat) {
        free_self();
        copy_base_matrix(&mat.matrix);
        return *this;
    }

    SuperLUMatrix& operator=(SuperLUMatrix&& mat) {
        free_self();
        std::memcpy(&matrix, &mat.matrix, sizeof(superlu_c::SuperMatrix));
        mat.matrix.Store = nullptr;
        return *this;
    }

    ~SuperLUMatrix() {
        free_self();
    }
};

template<typename T>
SuperLUMatrix<T> torch_to_superlu_mat(int rows, int cols,
                                      torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    const auto A_data_acc = A_data.accessor<T, 1>();
    const auto A_indices_acc = A_indices.accessor<int64_t, 1>();
    const auto A_indptr_acc = A_indptr.accessor<int64_t, 1>();

    T* A_data_ptr = new T[A_data.size(0)];
    superlu_c::int_t* A_indices_ptr = new superlu_c::int_t[A_indices.size(0)];
    superlu_c::int_t* A_indptr_ptr = new superlu_c::int_t[A_indptr.size(0)];

    for (uint64_t i = 0; i < A_data.size(0); i++) {
        A_data_ptr[i] = A_data_acc[i];
    }
    for (uint64_t i = 0; i < A_indices.size(0); i++) {
        A_indices_ptr[i] = A_indices_acc[i];
    }
    for (uint64_t i = 0; i < A_indptr.size(0); i++) {
        A_indptr_ptr[i] = A_indptr_acc[i];
    }

    superlu_c::int_t rows_int_t = static_cast<superlu_c::int_t>(rows);
    superlu_c::int_t cols_int_t = static_cast<superlu_c::int_t>(cols);

    SuperLUMatrix<T> mat;
    mat.matrix.nrow = rows;
    mat.matrix.ncol = cols;

    superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(mat.matrix.Store);
    store->nnz = A_data.size(0);
    store->nzval = static_cast<void*>(A_data_ptr);
    store->rowind = A_indices_ptr;
    store->colptr = A_indptr_ptr;

    return mat;
}

template<typename T>
std::vector<torch::Tensor> superlu_to_torch_mat(const SuperLUMatrix<T>& mat) {
    const int64_t rows = mat.matrix.nrow;
    const int64_t cols = mat.matrix.ncol;

    int64_t nnz;
    T* A_data;
    int64_t* A_indices;
    int64_t* A_indptr;

    /* Copy data over from the SuperLUMatrix */
    if (mat.matrix.Stype == superlu_c::SLU_NC) {
        superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(mat.matrix.Store);
        nnz = store->nnz;

        A_data = new T[nnz];
        A_indices = new int64_t[nnz];
        A_indptr = new int64_t[cols + 1];

        std::cout << "Num rows " << rows << std::endl;
        std::cout << "Num cols " << cols << std::endl;

        std::memcpy(A_data, store->nzval, nnz * sizeof(T));
        for (int64_t i = 0; i < nnz; i++) {
            A_indices[i] = static_cast<int64_t>(store->rowind[i]);
        }
        for (int64_t i = 0; i < cols + 1; i++) {
            A_indptr[i] = static_cast<int64_t>(store->colptr[i]);
        }
    } else if (mat.matrix.Stype == superlu_c::SLU_SC) {
        superlu_c::SCformat* store = static_cast<superlu_c::SCformat*>(mat.matrix.Store);
        /* TODO */
        throw std::runtime_error("Conversion from columnwise supernodal format not implemented.");
    }

    /* Return torch representations */
    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64);

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(type_to_torch_dtype(typeid(T)));

    torch::Tensor A_data_T = torch::from_blob(A_data, { nnz }, std::free, scalar_tens_opts);
    torch::Tensor A_indices_T = torch::from_blob(A_indices, { nnz }, std::free, int_tens_opts);
    torch::Tensor A_indptr_T = torch::from_blob(A_indptr, { cols }, std::free, int_tens_opts);

    return { A_data_T, A_indices_T, A_indptr_T };
}

template<typename T>
std::vector<torch::Tensor> superlu_to_torch_mat(SuperLUMatrix<T>&& mat) {
    const int64_t nnz = mat.matrix.Store.nnz;
    const int64_t rows = mat.matrix.nrow;
    const int64_t cols = mat.matrix.ncol;

    int64_t* A_indices = new int64_t[nnz];
    int64_t* A_indptr = new int64_t[cols + 1];

    for (int64_t i = 0; i < nnz; i++) {
        A_indices[i] = (int64_t) mat.matrix.Store.rowind[i];
    }
    for (int64_t i = 0; i < cols + 1; i++) {
        A_indptr[i] = (int64_t) mat.matrix.Store.colptr[i];
    }

    /* Move data over from the SuperLUMatrix */
    mat.Store.nzval = nullptr;
    std::free(mat.Store.rowind);
    mat.Store.rowind = nullptr;
    std::free(mat.Store.colptr);
    mat.Store.colptr = nullptr;
    mat.Store.nnz = 0;

    /* Return torch representations */
    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64);

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(type_to_torch_dtype(typeid(T)));

    torch::Tensor A_data_T = torch::from_blob(mat.matrix.Store.nzval, { nnz }, std::free, scalar_tens_opts);
    torch::Tensor A_indices_T = torch::from_blob(A_indices, { nnz }, std::free, int_tens_opts);
    torch::Tensor A_indptr_T = torch::from_blob(A_indptr, { cols }, std::free, int_tens_opts);

    return { A_data_T, A_indices_T, A_indptr_T };
}

template<typename T>
struct coo_coordinate_t {
    int64_t row, col;
    T val;

    bool operator<(const coo_coordinate_t<T>& other) const {
        if (col < other.col) {
            return true;
        } else if (col > other.col) {
            return false;
        } else return (row < other.row);
    }

    bool operator=(const coo_coordinate_t<T>& other) const {
        return (col == other.col && row == other.row);
    }

    bool operator>(const coo_coordinate_t<T>& other) const {
        return !(*this < other && *this == other);
    }

    template <typename Q>
    friend std::ostream& operator<<(std::ostream& os, const coo_coordinate_t<Q>& coord);
};

template<typename T>
std::ostream& operator<<(std::ostream& ostream, const coo_coordinate_t<T>& coord) {
    ostream << "(" << coord.row << ", " << coord.col << "): " << coord.val;
    return ostream;
}

template<typename T>
std::vector<torch::Tensor> superlu_factorize(SuperLUMatrix<T>& A) {
    superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(A.matrix.Store);
    const int64_t nnz = store->nnz;
    const int64_t rows = A.matrix.nrow;
    const int64_t cols = A.matrix.ncol;

    int32_t* row_perm = new int32_t[rows];
    int32_t* col_perm = new int32_t[cols];
    int32_t* elim_tree = new int32_t[cols];

    int32_t info;
    superlu_c::GlobalLU_t Glu;
    superlu_c::SuperLUStat_t stat;
    superlu_c::superlu_options_t options;
    superlu_c::set_default_options(&options);
    superlu_c::StatInit(&stat);

    /* First, get a column permutation using minimum col degree ordering */
    superlu_c::SuperMatrix AP;
    superlu_c::get_perm_c(3, &A.matrix, col_perm);
    superlu_c::sp_preorder(&options, &A.matrix, col_perm, elim_tree, &AP);

    const int32_t panel_size = superlu_c::sp_ienv(1);
    const int32_t relax = superlu_c::sp_ienv(2);

    /* Run the factorization */
    superlu_c::SuperMatrix L;
    superlu_c::SuperMatrix U;

    if (typeid(T) == typeid(float)) {
        superlu_c::sgstrf(&options, &AP, relax, panel_size, elim_tree, nullptr, 0, col_perm, row_perm,
               &L, &U, &Glu, &stat, &info);
    } else if (typeid(T) == typeid(double)) {
        superlu_c::dgstrf(&options, &AP, relax, panel_size, elim_tree, nullptr, 0, col_perm, row_perm,
               &L, &U, &Glu, &stat, &info);
    } else {
        throw std::runtime_error("Unknown type for SuperLU factorization.");
    }

    /* Check SuperLU's error flag */
    if (info > 0) {
        throw std::runtime_error("Failed to factorize matrix: factor U is exactly singular.");
    } else if (info < 0) {
        throw std::runtime_error("Illegal argument was passed to the SuperLU factorization.");
    }

    /** Extract LU factor */
    superlu_c::NCformat* Ustore = static_cast<superlu_c::NCformat*>(U.Store);
    superlu_c::SCformat* Lstore = static_cast<superlu_c::SCformat*>(L.Store);

    std::vector<coo_coordinate_t<T>> out_coo_format;
    out_coo_format.reserve(Lstore->nzval_colptr[cols]);

    const T eps = T(1e-16);

    /* What in the world is SuperLU doing here ????
     * The U factor is stored in both L and U ?!?!
     * We are following what SciPy is doing to convert the LU factorization
     * https://github.com/scipy/scipy/blob/e799ae4bb3c7c2d165e2fcb7bccb5373b91e267d/scipy/sparse/linalg/_dsolve/_superluobject.c#L553 */
    for (int64_t supernode = 0; supernode <= Lstore->nsuper; ++supernode) {
        const int64_t supernode_start_col = Lstore->sup_to_col[supernode];
        const int64_t supernode_end_col = Lstore->sup_to_col[supernode + 1];
        const int64_t idx_start = Lstore->rowind_colptr[supernode_start_col];
        const int64_t idx_end = Lstore->rowind_colptr[supernode_start_col + 1];

        for (int64_t col = supernode_start_col; col < supernode_end_col; ++col) {
            /* Extract from U */
            for (int64_t idx = Ustore->colptr[col]; idx < Ustore->colptr[col + 1]; ++idx) {
                const int64_t row = Ustore->rowind[idx];
                const T val = static_cast<T*>(Ustore->nzval)[idx];
                if (std::abs(val) > eps) {
                    out_coo_format.push_back({ row, col, val });
                }
            }

            /* Extract from L */
            T* src = &static_cast<T*>(Lstore->nzval)[Lstore->nzval_colptr[col]];
            for (int64_t idx = idx_start; idx < idx_end; ++idx) {
                const int64_t row = Lstore->rowind[idx];
                const T val = src[idx - idx_start];
                if (std::abs(val) > eps) {
                    out_coo_format.push_back({ row, col, val });
                }
            }
        }
    }
    // std::sort(out_coo_format.begin(), out_coo_format.end()); /* <-- this doesn't do anything. why? */
    // for (int64_t i = 0; i < out_coo_format.size(); i++) {
    //     std::cerr << out_coo_format[i] << std::endl;
    // }

    /** Convert to CSC representation */
    auto int_tens_opts = torch::TensorOptions()
        .dtype(torch::kInt64);

    auto scalar_tens_opts = torch::TensorOptions()
        .dtype(type_to_torch_dtype(typeid(T)));

    const int64_t out_nnz = out_coo_format.size();
    torch::Tensor M_data = torch::empty({out_nnz}, scalar_tens_opts);
    torch::Tensor M_rowindices = torch::empty({out_nnz}, int_tens_opts);
    torch::Tensor M_colptr = torch::zeros({cols + 1}, int_tens_opts);

    auto M_data_acc = M_data.accessor<T, 1>();
    auto M_rowindices_acc = M_rowindices.accessor<int64_t, 1>();
    auto M_colptr_acc = M_colptr.accessor<int64_t, 1>();

    /* First, get number of nonzero entries per column */
    for (int64_t i = 0; i < out_nnz; i++) {
        M_colptr_acc[out_coo_format[i].col] ++;
    }

    /* Take the cumulative sum to get colptr */
    int64_t cumsum = 0;
    for (int64_t i = 0; i < cols; i++) {
        const int64_t temp = M_colptr_acc[i];
        M_colptr_acc[i] = cumsum;
        cumsum += temp;
    }
    M_colptr_acc[cols] = cumsum;

    /* Now insert row indices and data values */
    for (int64_t i = 0; i < out_nnz; i++) {
        M_data_acc[i] = out_coo_format[i].val;
        M_rowindices_acc[i] = out_coo_format[i].row;
    }

    /** Get the row and column permutation vectors */
    torch::Tensor col_perm_T = torch::empty({cols}, int_tens_opts);
    torch::Tensor row_perm_T = torch::empty({rows}, int_tens_opts);

    auto col_perm_acc = col_perm_T.accessor<int64_t, 1>();
    auto row_perm_acc = row_perm_T.accessor<int64_t, 1>();

    assert(rows == cols);
    for (int64_t i = 0; i < rows; i++) {
        col_perm_acc[i] = col_perm[i];
        row_perm_acc[i] = row_perm[i];
    }

    /* Free scratch space */
    superlu_c::Destroy_CompCol_Permuted(&AP);
    superlu_c::Destroy_SuperNode_Matrix(&L);
    superlu_c::Destroy_CompCol_Matrix(&U);
    delete[] elim_tree;
    delete[] col_perm;
    delete[] row_perm;

    return {M_data, M_rowindices, M_colptr, col_perm_T, row_perm_T};
}

#endif
