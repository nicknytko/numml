#ifndef SUPERLU_HPP_
#define SUPERLU_HPP_

#include <vector>

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
}

const c10::ScalarType type_to_torch_dtype(const std::type_info& type) {
    if (type == typeid(double)) {
        return torch::kFloat64;
    } else if (type == typeid(float)) {
        return torch::kFloat32;
    }
}

#define delete_if_exists(x) do {                \
    if ((x) != nullptr) {                       \
        delete x;                               \
        x = nullptr;                            \
    }} while (0);

template<typename T>
class SuperLUMatrix {
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

    SuperLUMatrix(const superlu_c::SuperMatrix& mat) {
        std::memcpy(&matrix, &mat, sizeof(superlu_c::SuperMatrix));
    }

    SuperLUMatrix(superlu_c::SuperMatrix&& mat) {
        std::memcpy(&matrix, &mat, sizeof(superlu_c::SuperMatrix));
        mat.Store = nullptr;
    }

    ~SuperLUMatrix() {
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

            delete matrix.Store;
            matrix.Store = nullptr;
        }
    }
};

template<typename T>
SuperLUMatrix<T> torch_to_superlu_mat(uint64_t rows, uint64_t cols,
                                      torch::Tensor A_data, torch::Tensor A_indices, torch::Tensor A_indptr) {
    const T* A_data_ptr = A_data.contiguous().data_ptr<T>();
    const superlu_c::int_t* A_indices_ptr = A_indices.to(torch::kInt).contiguous().data_ptr<T>();
    const superlu_c::int_t* A_indptr_ptr = A_indptr.to(torch::kInt).contiguous().data_ptr<T>();

    superlu_c::int_t rows_int_t = static_cast<superlu_c::int_t>(rows);
    superlu_c::int_t cols_int_t = static_cast<superlu_c::int_t>(cols);

    SuperLUMatrix<T> mat;
    mat.matrix.nrow = rows;
    mat.matrix.ncol = cols;
    mat.matrix.Store.nnz = A_data.size(0);
    mat.matrix.Store.nzval = A_data_ptr;
    mat.matrix.Store.rowind = A_indices_ptr;
    mat.matrix.Store.colptr = A_indptr_ptr;

    return mat;
}

template<typename T>
std::vector<torch::Tensor> superlu_to_torch_mat(const SuperLUMatrix<T>& mat) {
    const int64_t nnz = mat.matrix.Store.nnz;
    const int64_t rows = mat.matrix.nrow;
    const int64_t cols = mat.matrix.ncol;

    T* A_data = new T[nnz];
    int64_t* A_indices = new int64_t[nnz];
    int64_t* A_indptr = new int64_t[cols];

    /* Copy data over from the SuperLUMatrix */
    if (mat.matrix.Stype == superlu_c::SLU_NC) {
        superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(mat.matrix.Store);
        std::memcpy(A_data, store->nzval, nnz * sizeof(T));
        for (int64_t i = 0; i < nnz; i++) {
            A_indices[i] = static_cast<int64_t>(store->rowind[i]);
        }
        for (int64_t i = 0; i < cols; i++) {
            A_indptr[i] = static_cast<int64_t>(store->colptr[i]);
        }
    } else if (mat.matrix.Stype == superlu_c::SLU_SC) {
        superlu_c::SCformat* store = static_cast<superlu_c::SCformat*>(mat.matrix.Store);
        /* TODO */
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
    int64_t* A_indptr = new int64_t[cols];

    for (int64_t i = 0; i < nnz; i++) {
        A_indices[i] = (int64_t) mat.matrix.Store.rowind[i];
    }
    for (int64_t i = 0; i < cols; i++) {
        A_indptr[i] = (int64_t) mat.matrix.Store.colptr[i];
    }

    /* Move data over from the SuperLUMatrix */
    mat.Store.nzval = nullptr;
    delete mat.Store.rowind;
    mat.Store.rowind = nullptr;
    delete mat.Store.colptr;
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
std::vector<SuperLUMatrix<T>> superlu_factorize(const SuperLUMatrix<T>& A) {
    const int64_t nnz = A.matrix.Store.nnz;
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

    /* First, get a column permutation using minimum col degree ordering */
    superlu_c::SuperMatrix AP;
    superlu_c::get_perm_c(3, A.mat, col_perm);
    superlu_c::sp_preorder(options, A, col_perm, elim_tree, &AP);

    int32_t panel_size = superlu_c::sp_ienv(1);
    int32_t relax = superlu_c::sp_ienv(2);

    /* Run the factorization */
    superlu_c::SuperMatrix L;
    superlu_c::SuperMatrix U;

    if (typeid(T) == typeid(float)) {
        superlu_c::sgstrf(&options, &AP, relax, panel_size, elim_tree, nullptr, 0, col_perm, row_perm,
               &L, &U, &Glu, &stat, &info);
    } else if (typeid(T) == typeid(double)) {
        superlu_c::dgstrf(&options, &AP, relax, panel_size, elim_tree, nullptr, 0, col_perm, row_perm,
               &L, &U, &Glu, &stat, &info);
    }

    if (info > 0) {
        throw std::runtime_error("Failed to factorize matrix: factor U is exactly singular.");
    } else if (info < 0) {
        throw std::runtime_error("Illegal argument was passed to the SuperLU factorization.");
    }

    /* Create row permutation matrix. */
    SuperLUMatrix<int64_t> P;
    superlu_c::NCformat* store = static_cast<superlu_c::NCformat*>(P.matrix.Store);
    store->nnz = rows;
    store->nzval = new int32_t[rows];
    store->rowind = new int32_t[rows];
    store->colptr = new int32_t[rows];

    for (int64_t i = 0; i < rows; i++) {
        static_cast<int32_t*>(store->nzval)[i] = 1;
        store->colptr[i] = i;
        store->rowind[i] = row_perm[i];
    }

    /* Free scratch space */
    delete elim_tree;
    delete col_perm;
    delete row_perm;

    return {
        P,
        SuperLUMatrix<T>(std::move(L)),
        SuperLUMatrix<T>(std::move(U))
    };
}

#endif
