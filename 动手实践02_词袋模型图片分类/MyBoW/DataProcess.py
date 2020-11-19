import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import ImageInfo
from matplotlib.ticker import MultipleLocator

def plotCM(classes, matrix, savename):
    '''
    绘制混淆矩阵
    :param classes: 图片标签名称
    :param matrix: 混淆矩阵
    :param savename: 混淆矩阵保存路径
    :return: None
    '''

    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    plt.switch_backend('agg')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        ax.text(i, i, str('%.2f' % (matrix[i, i] * 100)), va='center', ha='center', fontsize=6)
    ax.set_xticklabels([''] + classes, rotation=90)
    ax.set_yticklabels([''] + classes)
    plt.savefig(savename)

def getDatasetFromFile(path):

    '''
    读取数据库中图片，生成训练集和测试集特征点向量
    :param path: 数据库路径
    :return: 训练集特征点向量，测试集特征点向量

    '''
    dirs = os.listdir(path)

    imgTrainSet = []
    imgTestSet = []

    label = 0
    feature_set = np.float32([]).reshape(0, 128)
    for dir in dirs:
        files = os.listdir(path + "/" + dir)
        #files.remove("Thumbs.db")

        for i in range(150):
        # ----------------------write your code bellow----------------------

        # ----------------------write your code above----------------------

        for i in range(150, len(files)):
        # ----------------------write your code bellow----------------------

        # ----------------------write your code above----------------------
        label += 1

    np.save("feature_set.npy", feature_set)
    return imgTrainSet, imgTestSet

