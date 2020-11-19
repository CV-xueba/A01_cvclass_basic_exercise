import cv2
import numpy as np

class ImageInfo:
    def __init__(self, image, label):
        self.image = image
        self.height, self.width = image.shape[0:2]
        self.label = label
        self.descriptors = None
        self.keypoints = None

    def getImgFeature(self):
        '''
        提取图片 img 的 SIFT 特征点
        :param self.img: 输入图片的矩阵
        :return: img 的特征点向量，img 的 keypoints 列表
        '''
        # ----------------------write your code bellow----------------------

        # ----------------------write your code above----------------------
        self.descriptors = descriptors
        self.keypoints = keypoints
        return descriptors, keypoints

    def normalizeSIFT(self):
        '''
        归一化图片的 SIFT
        :param self.descriptors: 图片的特征点向量
        :return: None
        '''
        for i in range(len(self.descriptors)):
    # ----------------------write your code bellow----------------------

    # ----------------------write your code above----------------------