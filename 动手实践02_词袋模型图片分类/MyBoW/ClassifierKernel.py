import cv2
import numpy as np
import DataProcess
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import main

def trainSVM(data, labels):
    """
    训练 SVM 数据集。
    :param data: 输入图片投票向量
    :param labels: 图片数据集的标签
    :return: svm 模型
    """
    data = np.array(data, dtype='float32')
    labels = np.int64(labels)
    #----------------------write your code bellow----------------------

    # ----------------------write your code above----------------------
    return rbf_svc

def predict(data, model):
    """
    根据 svm 模型和数据集预测结果
    :param classes: 测试集数据集
    :param model: 输入 svm 模型
    :return: 数据集预测结果

    """
    data = np.array(data, dtype='float32')
    prediction = model.predict(data)

    return prediction

def evaluate(prediction, labels):
    '''
    评估模型预测结果
    :param prediction: 数据集预测结果
    :param labels: 数据集标签
    :return: 分类结果，混淆矩阵

    '''
    report = metrics.classification_report(labels, prediction, target_names=main.targetName)
    confuse_matrix = confusion_matrix(labels, prediction)
    return report, confuse_matrix