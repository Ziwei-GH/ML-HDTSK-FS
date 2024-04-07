from sklearn.metrics import hamming_loss, label_ranking_loss, coverage_error
from AP import *
from OE import *


def calculate_metrics(model_output, target_output):

    out_dim = target_output.shape[1]

    model_output = model_output.detach().numpy()
    model_label = (np.round(model_output) >= 1).astype(np.float32)
    target_output = target_output.detach().numpy()
    AP = average_precision(model_output.T, target_output.T)
    HL = hamming_loss(target_output, model_label)
    OE = One_Error(target_output, model_output)
    RL = label_ranking_loss(target_output, model_output)
    CV = (coverage_error(target_output, model_output) - 1) / out_dim

    return AP, HL, OE, RL, CV
