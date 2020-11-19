# Author: Ji Qiu （BUPT）
# cv_xueba@163.com

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math

class Vocabulary:
    def __init__(self, k):
        self.k = k
        self.vocabulary = None

    def generateBoW(self, path, random_state):
         """

        通过聚类算法，生成词袋模型

        :param path:词袋模型存储路径

        :param self.k: 词袋模型中视觉词汇的个数

        :param random_state: 随机数种子

        :return: 词袋模型视觉词汇矩阵

        """
        featureSet = np.load(path)
        np.random.shuffle(featureSet)
        kmeans = MiniBatchKMeans(n_clusters=self.k, random_state=random_state, batch_size=200).fit(featureSet)
        centers = kmeans.cluster_centers_
        self.vocabulary = centers
        np.save("Bow.npy", centers)
        return centers

    def getBow(self, path):
        """

        读取词袋模型文件

        :param path: 词袋模型文件路径

        :return: 词袋模型矩阵

        """
        centers = np.load(path)
        self.vocabulary = centers
        return centers

    def calSPMFeature(self, features, keypoints, center, img_x, img_y, numberOfBag):
        '''
        使用 SPM 算法，生成不同尺度下图片对视觉词汇的投票结果向量
        :param features:图片的特征点向量
        :param keypoints: 图片的特征点列表
        :param center: 词袋中的视觉词汇的向量
        :param img_x: 图片的宽度
        :param img_y: 图片的长度
        :param self.k: 词袋中视觉词汇的个数
        :return: 基于 SPM 思想生成的图片视觉词汇投票结果向量

        '''
        size = len(features)
        widthStep = math.ceil(img_x / 4)
        heightStep = math.ceil(img_y / 4)
        histogramOfLevelTwo = np.zeros((16, numberOfBag))
        histogramOfLevelOne = np.zeros((4, numberOfBag))
        histogramOfLevelZero = np.zeros((1, numberOfBag))
        for i in range(size):
            feature = features[i]
            keypoint = keypoints[i]
            x, y = keypoint.pt
            boundaryIndex = math.floor(x / widthStep) + math.floor(y / heightStep) * 4
            diff = np.tile(feature, (numberOfBag, 1)) - center
            SquareSum = np.sum(np.square(diff), axis=1)
            index = np.argmin(SquareSum)
            histogramOfLevelTwo[boundaryIndex][index] += 1

        for i in range(4):
            for j in range(4):
                histogramOfLevelOne[i] += histogramOfLevelTwo[j * 4 + i]

        for i in range(4):
            histogramOfLevelZero[0] += histogramOfLevelOne[i]

        result = np.float32([]).reshape(0, numberOfBag)
        result = np.append(result, histogramOfLevelZero * 0.25, axis=0)
        result = np.append(result, histogramOfLevelOne * 0.25, axis=0)
        result = np.append(result, histogramOfLevelTwo * 0.5, axis=0)
        return result

    def Imginfo2SVMdata(self, data):
        """

        将图片特征点数据转化为 SVM 训练的投票向量

        :param self.vocabulary: 词袋模型

        :param datas: 图片特征点数据

        :param self.k:

        词袋模型中视觉词汇的数量

        :return: 投票向量矩阵，图片标签

        """
        dataset = np.float32([]).reshape(0, self.k * 21)
        labels = []
        for simple in data:
            votes = self.calSPMFeature(simple.descriptors, simple.keypoints, self.vocabulary, simple.width, simple.height, self.k)
            votes = votes.ravel().reshape(1, self.k * 21)
            dataset = np.append(dataset, votes, axis=0)
            labels.append(simple.label)
        labels = np.array(labels)

        return dataset, labels

