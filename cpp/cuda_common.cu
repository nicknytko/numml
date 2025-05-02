#include "cuda_common.cuh"

/**
 * Given intermediate row, column, value COO representation,
 * sort entries first by row then by column.
 */
void lexsort_coo_ijv(torch::Tensor& Bhat_I,
                     torch::Tensor& Bhat_J,
                     torch::Tensor& Bhat_V) {

    at::cuda::CUDAStream main_stream = at::cuda::getCurrentCUDAStream();

    /* Sort first by columns... */
    torch::Tensor argsort = std::get<1>(Bhat_J.sort(false, -1, false));
    torch::Tensor i_temp = torch::empty_like(Bhat_I);
    torch::Tensor j_temp = torch::empty_like(Bhat_J);
    torch::Tensor v_temp = torch::empty_like(Bhat_V);

    const int64_t Bhat_total_nnz = Bhat_I.size(0);

    /* ...permute entries into their correct positions */
    AT_DISPATCH_FLOATING_TYPES(Bhat_V.scalar_type(), "lexsort_coo_ijv", [&] {
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_I, int64_t), tensor_acc(i_temp, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_J, int64_t), tensor_acc(j_temp, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(Bhat_V, scalar_t), tensor_acc(v_temp, scalar_t), tensor_acc(argsort, int64_t), true);
    });
    cuda_check_kernel_launch_err();

    /* Now, stable sort on rows.

       Rant incoming:
       Torch's (arg)sort interface is broke on 1.12.1, so we perform this awful incantation.
       Specifically, we can't even call argsort with stable, so we have to call
       sort and grab the second argument, making sure to reshape inputs because for some
       reason it breaks on flat vectors!! */
    argsort = std::get<1>(i_temp.reshape({1, -1}).sort(true, -1, false)).flatten();

    /* ...and again permute entries into correct spots */
    AT_DISPATCH_FLOATING_TYPES(Bhat_V.scalar_type(), "lexsort_coo_ijv", [&] {
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(i_temp, int64_t), tensor_acc(Bhat_I, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(j_temp, int64_t), tensor_acc(Bhat_J, int64_t), tensor_acc(argsort, int64_t), true);
        cuda_kernel_tensor_permute<<<(Bhat_total_nnz + threads_per_block - 1) / threads_per_block, threads_per_block, 0, main_stream>>>(
            Bhat_total_nnz, tensor_acc(v_temp, scalar_t), tensor_acc(Bhat_V, scalar_t), tensor_acc(argsort, int64_t), true);
    });
    cuda_check_kernel_launch_err();
}
