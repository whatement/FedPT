import os

import numpy as np
import torch
import math
import datetime
import random
import json

from sklearn import metrics


# 用于更新与记录随着次数n增大的平均值value，v为前n次平均值，x为最新一次输入值
class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


# 计算总体预测准确率 accuracy
def matrix_score(predict_y, img_y, num_classes, weight_data_np):

    matrix = metrics.confusion_matrix(img_y, predict_y, labels=range(num_classes))
    recall = np.zeros(num_classes)
    for cls in range(num_classes):
        if matrix[cls].sum() != 0:
            recall[cls] = matrix[cls][cls] / matrix[cls].sum()

    weight_pml = weight_data_np / weight_data_np.sum()
    pml_accuracy = recall * weight_pml
    weight_pmv = np.ceil(weight_pml) / np.count_nonzero(weight_pml)
    pmv_accuracy = recall * weight_pmv

    return matrix, pml_accuracy.sum(), pmv_accuracy.sum(), recall


# 计算总体预测精度 precision
def sum_precision_score(predict_y, img_y):
    precision = metrics.precision_score(img_y, predict_y)
    return precision


# 随机数种子初始化
def seed_setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
    torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
    np.random.seed(seed)  # numpy产生的随机数一致
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_equiangular_tight(n, n_dims):
    n = n-1
    points = torch.randn(n, n_dims)
    normalized_points = torch.div(points, torch.norm(points, dim=1, keepdim=True))
    centroid = torch.mean(normalized_points, dim=0)
    simplex = torch.cat((centroid.unsqueeze(0), normalized_points), dim=0)
    return simplex