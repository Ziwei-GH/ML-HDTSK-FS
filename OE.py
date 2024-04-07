import numpy as np


# 评价指标--OE
def One_Error(label, pred):
    N = len(label)
    label_index = []
    for i in range(N):
        index = np.where(label[i] == 1)[0]  # np.where(condition) 当条件成立时，where返回的是每个符合condition条件元素的元祖
        label_index.append(index)
    OneError = 0
    for i in range(N):
        if np.array(np.argmax(pred[i])) in label_index[i]:
            OneError += 0
        else:
            OneError += 1
    OneError = OneError * 1.0 / N

    return OneError
