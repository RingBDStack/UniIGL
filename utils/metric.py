import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score,balanced_accuracy_score,roc_auc_score

import numpy as np

# def acc(labels, output):
#     pred = output.cpu().max(1)[1].numpy()
#     labels = labels.cpu().numpy()
#     return accuracy_score(labels, pred)


# def wf1(labels, output):
#     pred = output.cpu().max(1)[1].numpy()
#     labels = labels.cpu().numpy()
#     return f1_score(labels, pred, average='weighted')


# def mf1(labels, output):
#     pred = output.cpu().max(1)[1].numpy()
#     labels = labels.cpu().numpy()
#     return f1_score(labels, pred, average='macro')

# def bacc(labels, output):
#     pred = output.cpu().max(1)[1].numpy()
#     labels = labels.cpu().numpy()
#     return balanced_accuracy_score(labels, pred)

# def auroc(labels, output):
#     labels = labels.cpu().numpy()  
#     output = output.cpu().detach().numpy()  

#     n_classes = output.shape[1]  
#     labels_binary = np.eye(n_classes)[labels]

#     auroc = roc_auc_score(labels_binary, output, multi_class='ovr', average='macro')
#     return auroc

def one_hot_to_labels(one_hot_labels): 
    return np.argmax(one_hot_labels, axis=1)

def check_and_convert_labels(labels):
    # 检查标签是否是 one-hot 格式（假设标签是二维的且每行的最大值为1）
    if labels.ndim == 2 and labels.shape[1] > 1:  # 检查是否为 one-hot 格式
        return one_hot_to_labels(labels)
    return labels  # 如果不是 one-hot，直接返回原标签

def acc(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = check_and_convert_labels(labels.cpu().numpy())  # 如果是 one-hot 标签，则转换
    return accuracy_score(labels, pred)

def wf1(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = check_and_convert_labels(labels.cpu().numpy())  # 如果是 one-hot 标签，则转换
    return f1_score(labels, pred, average='weighted')

def mf1(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = check_and_convert_labels(labels.cpu().numpy())  # 如果是 one-hot 标签，则转换
    return f1_score(labels, pred, average='macro')

def bacc(labels, output):
    pred = output.cpu().max(1)[1].numpy()
    labels = check_and_convert_labels(labels.cpu().numpy())  # 如果是 one-hot 标签，则转换
    return balanced_accuracy_score(labels, pred)

def auroc(labels, output):
    labels = labels.cpu().numpy()
    output = output.cpu().detach().numpy()

    n_classes = output.shape[1]
    labels_binary = check_and_convert_labels(labels)  # 如果是 one-hot 标签，则转换

    auroc = roc_auc_score(labels_binary, output, multi_class='ovr', average='macro')
    return auroc