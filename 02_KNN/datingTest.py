import numpy as np
from knn import knn
import sys
import os
sys.path.append(os.path.join(sys.path[0], "../Normalize"))
from normalize_utils import min_max_norm

def file2matrix(filename):
    # 打开文件
    fr = open(filename)
    # 读取文件所有内容
    arrayOlines = fr.readlines()
    # 得到文件行数
    numberOfLines = len(arrayOlines)
    # 返回的NumPy矩阵numberOfLines行，3列
    returnMat = np.zeros((numberOfLines, 3))
    # 创建分类标签向量
    classLabelVector = []
    # 行的索引值
    index = 0
    # 读取每一行
    for line in arrayOlines:
        # 去掉每一行首尾的空白符，例如'\n','\r','\t',' '
        line = line.strip()
        # 将每一行内容根据'\t'符进行切片,本例中一共有4列
        listFromLine = line.split('\t')
        # 将数据的前3列进行提取保存在returnMat矩阵中，也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        # 根据文本内容进行分类1：不喜欢；2：一般；3：喜欢
        if listFromLine[-1] == 'didntLike':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'smallDoses':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'largeDoses':
            classLabelVector.append(3)
        index += 1
    # 返回标签列向量以及特征矩阵
    return returnMat, classLabelVector


def visual_data(x:np.array, y:np.array,  label): # 根据x，y展示数据。
    import matplotlib.pyplot as plt
    import seaborn as sns
    #sns.set_palette("hls", 3)
    sns.scatterplot(x, y, hue=label, palette=sns.color_palette("husl", 3),style=label)
    plt.show()
    print("")

def datingTest(path):
    from sklearn.model_selection import train_test_split
    # 导入数据
    datingMat, datingLabels = file2matrix(path)
    # visual_data(datingMat[:, 0], datingMat[:, 1], datingLabels) # 展示数据
    datingMat = min_max_norm(datingMat) # min_max归一化
    train_x, test_x, train_y, test_y = train_test_split(datingMat, datingLabels) #训练、测试样本分类
    error = 0
    for i in range(test_x.shape[0]):
        prediction = knn(test_x[i], train_x, train_y, 3) # 通过knn进行预测
        if prediction != test_y[i]:
            error += 1
        print(f"the classifier came back with:{prediction}, ground truth:{test_y[i]}")
    print(f"the total error rate is {error/test_x.shape[0]}")
    

if __name__ == '__main__':
    
    datingTest("02_KNN/data/datingTestSet.txt")
    print("")