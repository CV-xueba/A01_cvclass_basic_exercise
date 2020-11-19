import cv2
import DataProcess
import os
import numpy as np
import random
import Vocabulary
import ClassifierKernel

Path = "dataset"
targetName = os.listdir(Path)
random_state = 3
random.seed(random_state)
numOfBag = 200
num_feature = 200

def main():
    if not os.path.exists("train_set.npy"):
    	# 提取数据集中的样本，并划分训练集和测试集 
        trainData, testData = DataProcess.getDatasetFromFile(Path)
        #特征提取与词典生成
        vocabulary = Vocabulary.Vocabulary(numOfBag)
        vocabulary.generateBoW("feature_set.npy", random_state)
        # vocabulary.getBow("Bow.npy")
        # 图片表示,使用SPM方法生成图片特征向量
        trainSet, trainlabels = vocabulary.Imginfo2SVMdata(trainData)
        testSet, testlabels = vocabulary.Imginfo2SVMdata(testData)
        np.save("train_set.npy", trainSet)
        np.save("train_label.npy", trainlabels)
        np.save("test_set.npy", testSet)
        np.save("test_label.npy", testlabels)
    else:
        trainSet = np.load("train_set.npy")
        trainlabels = np.load("train_label.npy")
        testSet = np.load("test_set.npy")
        testlabels = np.load("test_label.npy")
        
    # 执行分类，使用支持向量机模型对测试集图片的类别进行预测 
    svm = ClassifierKernel.trainSVM(trainSet, trainlabels)
    result = ClassifierKernel.predict(trainSet, svm)
    testResult = ClassifierKernel.predict(testSet, svm)

    # 评估预测结果，并生成分类报告和输出混淆矩阵
    trainReport, _ = ClassifierKernel.evaluate(result, trainlabels)
    testReport, cm = ClassifierKernel.evaluate(testResult, testlabels)

    DataProcess.plotCM(targetName, cm, "confusion_matrix.png")
    print("训练集混淆矩阵：")
    print(trainReport)
    print("测试集混淆矩阵：")
    print(testReport)

if __name__ == '__main__':
    main()