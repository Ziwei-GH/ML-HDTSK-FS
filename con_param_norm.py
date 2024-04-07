import torch


def con_param_norm(con_param, in_dim, out_dim):

    mat = torch.split(con_param.clone(), in_dim + 1, dim=0)
    mat_list = [i.unsqueeze(0) for i in mat]
    mat_th = torch.cat(mat_list, dim=0)
    mat_norm = torch.norm(mat_th, p=2, dim=0)
    P_new = torch.sum(mat_norm)

    return P_new
