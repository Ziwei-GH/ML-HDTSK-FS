import pandas as pd
from input_g import *
from binary_difference import *
from norm_grad import *
from con_param_norm import *
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from calculate_metrics import *
from sklearn.metrics.pairwise import cosine_similarity
import random
import warnings
import os
warnings.filterwarnings("ignore")

rootpath = os.getcwd()


def train(in_dim, g, alpha, beta, target_output, gamma, max_iter, th_cor, th_r):

    target_output = target_output.double()
    out_dim = target_output.shape[1]

    # 初始化后件参数矩阵
    g_dim = g.shape[1]
    first = torch.inverse(g.T @ g + gamma * torch.eye(g_dim))
    con_param = first @ g.T @ target_output
    con_param_temp = con_param

    # 计算二值差分矩阵
    R, C = binary_difference(out_dim=out_dim, target_output=target_output, th_cor=th_cor)

    # 计算利普希茨常数
    Lip = torch.sqrt(2 * (torch.norm(g.T @ g) ** 2 + torch.norm(alpha * (R + C - 1)) ** 2))

    b = torch.tensor(1.0)
    b_temp = torch.tensor(1.0)
    loss_his = torch.zeros(max_iter)

    for t in range(max_iter):

        grad = norm_grad(con_param, in_dim, out_dim, num_fset, th_r)
        point = con_param + torch.divide((b_temp - 1), b) * (con_param - con_param_temp)
        grad_fp = g.T @ g @ point - g.T @ target_output + alpha * (point @ (R + C - 1)) + beta * grad

        b_temp = b
        b = torch.divide(1 + torch.sqrt(4 * b_temp ** 2 + 1), 2)

        con_param_temp = con_param.clone()
        con_param = point - (1 / Lip) * grad_fp

        model_output_temp = g @ con_param   # [num_sam,out_dim]

        # 计算损失
        loss = my_loss_fun(con_param, model_output_temp, target_output, R, C, alpha, beta)
        loss_his[t] = loss
        print('第{}次迭代，训练集损失值为：{:.4f}'.format(t, loss.data))
        if t >= 1:
            if loss_his[t-1] - loss_his[t] <= 1.0e-6:
                break
            elif loss <= 0:
                break
    model_output = g @ con_param

    # 评价指标
    AP, HL, OE, RL, CV = calculate_metrics(model_output=model_output, target_output=target_output)

    return con_param, AP, HL, OE, RL, CV


def my_loss_fun(con_param, model_output, target_output, R, C, alpha, beta):

    out_dim = target_output.shape[1]

    P_new = con_param_norm(con_param, in_dim, out_dim) / target_output.size(0)
    square_loss = ((model_output - target_output) ** 2).sum() / (2 * model_output.size(0))
    corr_loss = (R @ cosine_similarity(con_param.T)
                 + 1 - cosine_similarity(con_param.T) - C
                 + C @ cosine_similarity(con_param.T)).trace() / (2 * target_output.size(0))
    loss = square_loss + beta*P_new + alpha*corr_loss

    return loss


def test(g, target_output, con_param):

    target_output = target_output.double()

    model_output = g @ con_param

    AP, HL, OE, RL, CV = calculate_metrics(model_output=model_output, target_output=target_output)

    return model_output, AP, HL, OE, RL, CV


def set_random_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    set_random_seed(10)

    num_fea = 1449
    num_label = 45 
    dataset = pd.read_csv(rootpath + "/medical.csv", index_col=0)

    sam = torch.tensor(np.array(dataset.iloc[:, :num_fea]))
    label = torch.tensor(np.array(dataset.iloc[:, num_fea:]))

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    sam = torch.Tensor(min_max_scaler.fit_transform(sam))

    in_dim = sam.shape[1]
    m, maxiter, error, th_cor, th_r, power_n = 1.01, 100, 1.0e-6, 0.5, 0.01, 1
    num_fset, alpha, beta, gamma = 2, 0.1, 1, 1

    data = x_g(in_dim=in_dim, num_fset=num_fset, model_input=sam, m=m, c=num_fset,
               error=error, maxiter=maxiter, power_n=power_n)

    n_run = 5
    tra_AP_k_fold = torch.zeros(n_run)
    tra_HL_k_fold = torch.zeros(n_run)
    tra_OE_k_fold = torch.zeros(n_run)
    tra_RL_k_fold = torch.zeros(n_run)
    tra_CV_k_fold = torch.zeros(n_run)
    test_AP_k_fold = torch.zeros(n_run)
    test_HL_k_fold = torch.zeros(n_run)
    test_OE_k_fold = torch.zeros(n_run)
    test_RL_k_fold = torch.zeros(n_run)
    test_CV_k_fold = torch.zeros(n_run)

    max_iter = 100

    for i in range(n_run):

        # 划分训练集与测试集
        tra_sam, test_sam, tra_label, test_label = train_test_split(data, label, train_size=0.8)

        # 训练实例化
        con_param, AP, HL, OE, RL, CV = train(in_dim=in_dim, g=tra_sam, alpha=alpha, beta=beta,
                                              target_output=tra_label, gamma=gamma, max_iter=max_iter,
                                              th_cor=th_cor, th_r=th_r)

        # 测试实例化
        test_model_output, test_AP, test_HL, test_OE, test_RL, test_CV \
            = test(g=test_sam, target_output=test_label, con_param=con_param)

        # 训练集评价指标
        print('第{}轮, 训练集AP: {:.4f}, 测试集AP: {:.4f}'.format(i + 1, AP, test_AP))
        print('第{}轮, 训练集HL: {:.4f}, 测试集HL: {:.4f}'.format(i + 1, HL, test_HL))
        print('第{}轮, 训练集OE: {:.4f}, 测试集OE: {:.4f}'.format(i + 1, OE, test_OE))
        print('第{}轮, 训练集RL: {:.4f}, 测试集RL: {:.4f}'.format(i + 1, RL, test_RL))
        print('第{}轮, 训练集CV: {:.4f}, 测试集CV: {:.4f}'.format(i + 1, CV, test_CV))

        # 记录精度
        tra_AP_k_fold[i] = AP
        tra_HL_k_fold[i] = HL
        tra_OE_k_fold[i] = OE
        tra_RL_k_fold[i] = RL
        tra_CV_k_fold[i] = CV
        test_AP_k_fold[i] = test_AP
        test_HL_k_fold[i] = test_HL
        test_OE_k_fold[i] = test_OE
        test_RL_k_fold[i] = test_RL
        test_CV_k_fold[i] = test_CV

    print('训练集平均精度: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
          .format(torch.mean(tra_AP_k_fold), torch.mean(tra_HL_k_fold), torch.mean(tra_OE_k_fold),
                  torch.mean(tra_RL_k_fold), torch.mean(tra_CV_k_fold)))
    print('测试集平均精度: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
          .format(torch.mean(test_AP_k_fold), torch.mean(test_HL_k_fold), torch.mean(test_OE_k_fold),
                  torch.mean(test_RL_k_fold), torch.mean(test_CV_k_fold)))
    print('测试集精度标准差: {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}'
          .format(np.std(np.array(test_AP_k_fold), ddof=1), np.std(np.array(test_HL_k_fold), ddof=1),
                  np.std(np.array(test_OE_k_fold), ddof=1), np.std(np.array(test_RL_k_fold), ddof=1),
                  np.std(np.array(test_CV_k_fold), ddof=1)))
    print('---------------------------------------------------------------')
    # 将结果保存到csv文件中
    with open(rootpath+'/medical_result.csv', 'a') as f:
        f.write(str(num_fset) + ',' + str(alpha) + ',' + str(beta) + ',' + str(gamma) + ','
                + str(torch.mean(tra_AP_k_fold)) + ',' + str(torch.mean(tra_HL_k_fold)) + ','
                + str(torch.mean(tra_OE_k_fold)) + ',' + str(torch.mean(tra_RL_k_fold)) + ','
                + str(torch.mean(tra_CV_k_fold)) + ',' + str(torch.mean(test_AP_k_fold)) + ','
                + str(torch.mean(test_HL_k_fold)) + ',' + str(torch.mean(test_OE_k_fold)) + ','
                + str(torch.mean(test_RL_k_fold)) + ',' + str(torch.mean(test_CV_k_fold)) + ','
                + str(np.std(np.array(test_AP_k_fold), ddof=1)) + ','
                + str(np.std(np.array(test_HL_k_fold), ddof=1)) + ','
                + str(np.std(np.array(test_OE_k_fold), ddof=1)) + ','
                + str(np.std(np.array(test_RL_k_fold), ddof=1)) + ','
                + str(np.std(np.array(test_CV_k_fold), ddof=1)) + '\n')
    f.close()
