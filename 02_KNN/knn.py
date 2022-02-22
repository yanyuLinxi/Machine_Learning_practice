from typing import List
import numpy as np
import collections

def createDataset():
    group = np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def knn(feature:np.array, dataset:np.array, labels:List[str], k:int):
    # 首先计算feature和各个x之间的欧式距离。两个向量按位平方和开根号。然后进行排序、投票。取距离最近的k个节点获取label
    # dataset shape: N, E. N=Num of sample, E:Embedding size
    # 按位平方和计算距离。
    feature = np.tile(feature, (dataset.shape[0], 1)) # shape N, E
    feature -= dataset # x-y
    feature **= 2 # (x-y)^2
    feature = feature.sum(axis=1) ** 0.5 # (\sum (x-y)^2)^(1/2)
    sorted_index = feature.argsort() # argsort意思是排序后，获取排序的值在原位置的下标，也是递增排序。这里求最近的k个邻居。
    classCount = collections.Counter()
    for i in range(k):
        classCount[labels[sorted_index[i]]] += 1
    sorted_label = sorted(classCount.items(), key=lambda x:x[1], reverse=True) # 获取字典的所有键值并排序。排序获取投票最多的值，注意sort是递增排序。
    return sorted_label[0][0]
    
if __name__ == '__main__':
    g, l = createDataset()
    #g = np.array([[1.0, 1.0], [0., 0.],[0.5, 0]])
    label = knn(np.array([0., 0.]), g, l , 3)
    print("label",label)
    print("")