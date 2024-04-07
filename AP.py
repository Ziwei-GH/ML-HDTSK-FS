import numpy as np


def average_precision(outputs, test_target):

    num_class, num_instance = test_target.shape

    temp_outputs = []
    temp_test_target = []

    for i in range(num_instance):
        temp = test_target[:, i]
        if (np.sum(temp) != num_class) and (np.sum(temp) != -num_class):
            temp_outputs.append(outputs[:, i])
            temp_test_target.append(temp)

    outputs = np.array(temp_outputs).T
    test_target = np.array(temp_test_target).T

    num_class, num_instance = outputs.shape

    label = []
    not_label = []
    label_size = np.zeros((1, num_instance))

    for i in range(num_instance):
        temp = test_target[:, i]
        label_size[0, i] = np.sum(temp == 1)

        label_i = []
        not_label_i = []
        for j in range(num_class):
            if temp[j] == 1:
                label_i.append(j + 1)
            else:
                not_label_i.append(j + 1)

        label.append(label_i)
        not_label.append(not_label_i)

    aveprec = 0

    for i in range(num_instance):
        temp = outputs[:, i]
        index = np.argsort(temp)
        index += 1

        indicator = np.zeros(num_class)

        for m in range(int(label_size[0, i])):

            loc = np.where(np.array(label[i][m]) == index)

            indicator[loc] = 1

        summary = 0

        for m in range(int(label_size[0, i])):
            loc = np.where(np.array(label[i][m]) == index)
            summary += np.sum(indicator[loc[0][0]:num_class]) / (num_class - loc[0][0])

        a = label_size[0, i]
        if a == 0:
            a = 1

        aveprec += summary / a

    average_precision = aveprec / num_instance

    return average_precision
