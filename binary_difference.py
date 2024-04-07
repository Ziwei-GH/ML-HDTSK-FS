import torch


def binary_difference(out_dim, target_output, th_cor):

    B = torch.empty((out_dim, out_dim)).double()

    out_dim2 = 0

    for bi in range(out_dim):
        for bj in range(out_dim2, out_dim):

            target_output_sum_this = target_output[:, bi]+target_output[:, bj]
            num_zero_one = target_output_sum_this[target_output_sum_this == 1].shape[0]
            num_one_one = target_output_sum_this[target_output_sum_this == 2].shape[0]

            if int(num_zero_one+num_one_one) == 0:
                B[bi, bj] = 0
            else:
                B[bi, bj] = num_zero_one/(num_one_one + num_zero_one)

            B[bj, bi] = B[bi, bj]

        out_dim2 = int(out_dim2+1)

    R = B.clone()
    C = B.clone()
    R[R <= th_cor] = 0
    C[C >= th_cor] = 1
    C = C.fill_diagonal_(1)

    return R, C
