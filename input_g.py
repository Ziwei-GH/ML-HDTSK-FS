import torch
from skfuzzy.cluster import cmeans


def x_g(in_dim, num_fset, model_input, m, c, error, maxiter, power_n):

    n_examples = model_input.shape[0]
    model_input = model_input.double()

    # 计算隶属度值
    center, _, _, _, _, _, _ = cmeans(model_input.T, c=c, m=m, error=error, maxiter=maxiter, init=None)
    center = torch.tensor(center)

    '''
    计算激活强度
    '''
    x_e = torch.cat([model_input, torch.ones(n_examples, 1)], 1)
    wt = torch.zeros((n_examples, num_fset), dtype=torch.float64)
    for i in range(num_fset):
        v1 = torch.tile(center[i, :], (n_examples, 1))
        b = torch.sqrt(in_dim / torch.tensor(1490)).double()
        bb = torch.full((n_examples, in_dim), b)
        wt[:, i] = torch.exp(-torch.sum(((model_input - v1) ** 2) ** power_n / (2 * bb ** 2), dim=1))

    wt2 = torch.sum(wt, dim=1)
    ss = wt2 == 0
    wt2[ss] = torch.finfo(torch.float32).eps
    wt = wt / torch.tile(wt2.reshape((-1, 1)), (1, num_fset))

    x_g = torch.zeros((n_examples, num_fset * (in_dim + 1)), dtype=torch.float64)
    for i in range(num_fset):
        wt1 = wt[:, i]
        wt2 = torch.tile(wt1.reshape((-1, 1)), (1, in_dim + 1))
        x_g[:, (i * (in_dim + 1)):((i + 1) * (in_dim + 1))] = x_e * wt2

    return x_g
