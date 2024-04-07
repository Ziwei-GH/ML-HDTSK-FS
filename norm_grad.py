import torch


def norm_grad(con_param, in_dim, out_dim, num_fset, th_r):

    mat = torch.split(con_param.clone(), in_dim + 1, dim=0)
    mat_list = [i.unsqueeze(0) for i in mat]
    mat_th = torch.cat(mat_list, dim=0)
    mat_norm = torch.norm(mat_th, p=2, dim=0)

    # 找到大于阈值的元素的位置
    mask = mat_norm <= th_r

    # 将这些元素除以它们自身的平方
    mat_norm[mask] = (mat_norm[mask] * mat_norm[mask]) / (2 * th_r) + th_r / 2

    mat_th = mat_th/mat_norm
    mat_th_out = mat_th.view(num_fset * (in_dim + 1), out_dim)

    return mat_th_out
