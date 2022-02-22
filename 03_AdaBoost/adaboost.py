# -*- coding: utf-8 -*-
from sqlite3 import paramstyle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

def loadDataSet(fileName):
    """加载文件

    Args:
        fileName (_type_): 文件名

    Returns:
        dataMat: 数据矩阵
        labelMat: 数据标签
    """        
    
    # 特征个数
    numFeat = len((open(fileName).readline().split('\t')))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        # 将Label列剔除，只存储特征矩阵
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    #data = pd.dataFrame()
    return dataMat, labelMat



def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """根据数据矩阵进行分类，返回预测值。

    Args:
        dataMatrix (_type_): 样本数据
        dimen (_type_): 样本列
        threshVal (_type_): 样本阈值
        threshIneq (_type_): 是小于阈值还是大于阈值（判负）

    Returns:
        _type_: 预测值
    """    
    # 初始化retArray为全1列向量
    retArray = np.ones((np.shape(dataMatrix)[0], 1)) 
    # 阈值之外为正。阈值之内为负
    # 阈值之外标记为-1，阈值之内标记为1
    # less than
    if threshIneq == 'lt':
        # 如果小于阈值则赋值为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    # greater than
    else:
        # 如果大于阈值则赋值为-1
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray



def buildStump(dataArr, classLabels, D):
    """获取数据，对每一列特征选择numSteps个分裂点进行计算。选取误差最小的列和分裂点。建立单层决策树。

    Args:
        dataArr (_type_): 样本数据
        classLabels (_type_): 样本标签
        D (_type_): 上一轮样本权重

    Returns:
        bestStump - 最佳单层决策树信息
        minError - 最小误差
        bestClassEst - 最佳的分类结果
    """    
    # 输入数据转为矩阵(5, 2)
    dataMatrix = np.mat(dataArr)
    # 将标签矩阵进行转置(5, 1)
    labelMat = np.mat(classLabels).T
    m, n = np.shape(dataMatrix)
    numSteps = 10.0 # 
    bestStump = {}
    # (5, 1)全零列矩阵
    bestClasEst = np.mat(np.zeros((m, 1)))
    # 最小误差初始化为正无穷大inf
    minError = float('inf')
    # 遍历所有特征
    for i in range(n):
        # 找到(每列)特征中的最小值和最大值
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        # 计算步长
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            # 大于和小于的情况均遍历，lt:Less than  gt:greater than
            for inequal in ['lt', 'gt']:
                # 计算阈值
                threshVal = (rangeMin + float(j) * stepSize)
                # 计算分类结果
                # 这个阈值就是当前特征的分裂点。由于是二分裂，所以直接学习将左右节点分别判做负值的误差率。
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal) 
                # 初始化误差矩阵
                errArr = np.mat(np.ones((m, 1)))
                # 分类正确的，赋值为0
                errArr[predictedVals == labelMat] = 0
                # 计算误差。计算的时候误差需要乘以样本的权重。
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                # 找到误差最小的分类方式
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    # 第i个特征
                    bestStump['dim'] = i
                    # 此时的阈值
                    bestStump['thresh'] = threshVal
                    # 分类方式是lt还是gt
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst



def adaBoostTrainDS(dataArr, classLabels, numIt=60):
    """
    构建numIt个单层决策树。然后存储每个决策树的权重（根据前面轮数的权重）
    Parameters:
        dataArr - 数据矩阵
        classLabels - 数据标签
        numIt - 最大迭代次数
    
    Returns:
        weakClassArr - 存储单层决策树的list
        aggClassEsc - 训练的label
    """
    weakClassArr = []
    # 获取数据集的行数
    m = np.shape(dataArr)[0]
    # 样本权重，每个样本权重相等，即1/n
    D = np.mat(np.ones((m, 1)) / m)
    # 初始化为全零列
    aggClassEst = np.mat(np.zeros((m, 1))) # TODO: 这是什么
    # 迭代
    for i in range(numIt):
        # 构建单层决策树
        # 最佳分裂，误差，分类结果。
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        # print("D:", D.T)
        # 计算弱学习算法权重alpha，使error不等于0，因为分母不能为0
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        # 存储弱学习算法权重
        bestStump['alpha'] = alpha
        # 存储单层决策树
        weakClassArr.append(bestStump)
        # 打印最佳分类结果
        # print("classEst: ", classEst.T)
        # 计算e的指数项
        # 根据分类结果的正确性，计算$e^{\alpha -y f(x)}$
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
        # 计算递推公式的分子
        D = np.multiply(D, np.exp(expon))
        # 根据样本权重公式，更新样本权重
        D = D / D.sum()
        # 计算AdaBoost误差，当误差为0的时候，退出循环
        # 以下为错误率累加计算
        # 累加的分类结果。
        aggClassEst += alpha * classEst
        # print("aggClassEst: ", aggClassEst.T)
        # 计算误差
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        # 计算误差率
        errorRate = aggErrors.sum() / m
        # print("total error:", errorRate)
        if errorRate == 0.0:
            # 误差为0退出循环
            break
    return weakClassArr, aggClassEst



def plotROC(predStrengths, classLabels):
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 检测光标的位置
    cur = (1.0, 1.0)
    # 用于计算AUC
    ySum = 0.0
    # 统计正类的数量
    numPosClas = np.sum(np.array(classLabels) == 1.0)
    # y轴步长
    yStep = 1 / float(numPosClas)
    # x轴步长
    xStep = 1 / float(len(classLabels) - numPosClas)
    # 预测强度顺序  argsort函数返回的是数组值从小到大的索引值
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    # 把画布分成1*1的格子。把图形放在第1格
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        # 多一个TP向y轴移动一步
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        # 多一个FP向x轴移动一步
        else:
            delX = xStep
            delY = 0
            # 高度累加
            ySum += cur[1]
        # 绘制ROC
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        # 更新绘制光标位置
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('AdaBoost马疝病检测系统的ROC曲线', FontProperties=font)
    plt.xlabel('假阳率', FontProperties=font)
    plt.ylabel('真阳率', FontProperties=font)
    ax.axis([0, 1, 0, 1])
    # 计算AUC
    print('AUC面积为：', ySum * xStep)
    plt.show()


if __name__ == '__main__':
    dataArr, LabelArr = loadDataSet('03_AdaBoost/data/horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, LabelArr)
    plotROC(aggClassEst.T, LabelArr)
    print("")
    