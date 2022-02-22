import numpy as np
import random


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))



def gradAscent(dataMath, classLabels):
    # 进行一次梯度上升。梯度上升寻找最大值。公式为Wights = Wights + \alpha * \delta f(x)
    # 这里的损失函数是没有取负值的，所以是梯度上升。
    # 转换成numpy的mat(矩阵)
    dataMatrix = np.mat(dataMath)
    # 转换成numpy的mat(矩阵)并进行转置
    labelMat = np.mat(classLabels).transpose()
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 移动步长，也就是学习效率，控制更新的幅度
    alpha = 0.01
    # 最大迭代次数
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        # 梯度上升矢量化公式
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    # 将矩阵转换为数组，返回权重数组
    # mat.getA()将自身矩阵变量转化为ndarray类型变量
    return weights.getA()



def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    # 返回dataMatrix的大小，m为行数，n为列数
    m, n = np.shape(dataMatrix)
    # 参数初始化
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            # 每次都降低alpha的大小
            alpha = 4/(1.0+j+i)+0.01
            # 随机选择样本
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选择一个样本计算h
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            # 计算误差
            error = classLabels[randIndex] - h
            # 更新回归系数
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 删除已使用的样本
            del(dataIndex[randIndex])
    # 返回
    return weights



def colicTest():
    # 打开训练集
    frTrain = open('05_LogisticRegression/data/horseColicTraining.txt')
    # 打开测试集
    frTest = open('05_LogisticRegression/data/horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        # trainingLabels.append(lineArr)
        trainingLabels.append(float(currLine[-1]))
    # 使用改进的随机上升梯度训练
    # trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    # 使用上升梯度训练
    trainWeights = gradAscent(np.array(trainingSet), trainingLabels)
    errorCount = 0
    numTestVect = 0.0
    for line in frTest.readlines():
        numTestVect += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        # if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[-1]):
        if int(classifyVector(np.array(lineArr), trainWeights[:,0])) != int(currLine[-1]):
            errorCount += 1
    # 错误概率计算
    errorRate = (float(errorCount) / numTestVect) * 100
    print("测试集错误率为：%.2f%%" % errorRate)
    
    
"""
函数说明：分类函数

Parameters:
    inX - 特征向量
    weights - 回归系数
    
Returns:
    分类结果

Modify:
    2018-07-22
"""
def classifyVector(inX, weights):
    # 判定函数||(x), 取阈值为0.5
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


if __name__ == '__main__':
    colicTest()
    print("")