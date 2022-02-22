# 建立一颗cart决策树
import numpy as np
import pandas as pd
from typing import List, Union, Dict, Tuple

class TreeNode:
    def __init__(self,  val=None, col=None, left=None, right=None):
        self.val = val # 当这个节点是叶子节点的时候，这个值存储的是叶子节点的均值，当这个节点是非叶子节点时，存储的是分裂值
        self.col = col
        self.left = left
        self.right = right
        

class CART:
    def __init__(self, min_impurity_decrease = 0, min_samples_leaf = 1):
        self.criterion = "mse"
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
    
    
    def fit(self, x:Union[pd.DataFrame,List], y) -> None:
        # 伪代码：
        # 处理输入。
        # createTree
        # 如果y里所有的值都相同，不分裂。 
        # 对特征循环：
        #   chooseBestSplit
        #   获取该列特征所有值的排序，获取每两个元素的中位数作为分裂点。
        #   对于每一个分裂点：
        #       将数据划分成两列
        #       使用criterion计算两列的误差。
        #       更新最小的误差和划分点。
        # 比较所有特征，获取最佳特征划分特征和最佳的划分点。
        # 如果误差小于min_impurity_decrease，则不分裂，返回leafType作为值。
        # 如果分裂出的叶子节点的数量小于min_samples_leaf，则不分裂，返回leafType作为均值
        try:
            if type(x) != pd.DataFrame:
                x = pd.DataFrame(x)
            if type(y) not in [pd.DataFrame, pd.Series]:
                y = pd.Series(y)
            assert type(x) == pd.DataFrame, "dataset的数据类型应该为pandas.DataFrame或者可以转换成DataFrame"
        except Exception:
            raise Exception #不解决异常，直接抛出。
        
        assert len(x) == len(y), "数据的长度和标签长度不一致"
        self._sample_nums = len(x)
        self.decision_tree_ = self.__build_tree(x, y)
        
    
    def prediction(self, x:pd.DataFrame) -> float:
        # 如何进行预测？
        # 将x按照决策树要求进行划分。
        # 获取当前节点的分裂col、分裂值，进行判断送入左子树还是右子树。
        # 当col是空的时候，返回val
        cur_node = self.decision_tree_
        
        while cur_node.col is not None:
            col, val = cur_node.col, cur_node.val
            if x[col] < val:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return cur_node.val
        
    def prune(self, x, y):
        # 使用测试集数据对生成的decision Tree进行剪枝。
        # 递归的调用子函数进行剪枝。
        # 首先判断x、y非空。如果测试集为空，将树做塌陷处理。
        #   递归的将左右子树的均值替换为子树的均值。
        #   并设置节点的col为None。
        # 然后将x、y根据决策树要求进行划分。
        # 然后递归判断左右子树是否可以剪枝
        # 当左右子树都叶子节点时，则可以对当前节点进行剪枝。
        #   计算测试集上，左子树的误差减去左子树的均值 的平方 加上 右子树的。 减去 当前节点的测试集
        #   if (l-l_mean)^2 + (r-r_mean)^2 < (x-(l_mean+r_mean)/2)^2
        #       剪枝
        #   else: 返回当前树
        # 返回当前节点。
        try:
            if type(x) != pd.DataFrame:
                x = pd.DataFrame(x)
            if type(y) not in [pd.DataFrame, pd.Series]:
                y = pd.Series(y)
            assert type(x) == pd.DataFrame, "dataset的数据类型应该为pandas.DataFrame或者可以转换成DataFrame"
        except Exception:
            raise Exception #不解决异常，直接抛出。
        
        print(self.get_decision_tree_dict())
        self.decision_tree_ = self.__prune_sub_tree(self.decision_tree_, x, y)
        print(self.get_decision_tree_dict())
        print("")
    
    
    def get_decision_tree_dict(self):
        
        def traversal(root):
            # 进行遍历这棵树。当遍历左子树，遍历右子树。后序遍历
            # 当这个节点为None时返回
            # 当左子树、右子树都为None的时候，返回当前的val
            # 否则构建左子树，构建右子树。
            if root.col is None:
                return root.val
            left = traversal(root.left)
            right = traversal(root.right)
            return dict(spFeature=root.col, spVal=root.val, left=left, right=right)
        return traversal(self.decision_tree_)
          
    def __mean_sub_tree(self, node) -> TreeNode:
        # 递归平均当前节点。将这个树替换为叶子节点。
        # 当当前节点col为None时，返回val值
        # 否则当前节点不是叶子节点
        #   平均左子树，
        #   平均右子树
        #   获取左子树的val、右子树的val
        #   设置左右子树为None
        #   设置当前值为左右子树的val的均值
        #   设置col为None
        # 返回当前节点。
        if node.col is None:
            return node
        
        # 递归平均左右子树
        node.left = self.__mean_sub_tree(node.left)
        node.right = self.__mean_sub_tree(node.right)
        
        node.val = (node.left.val + node.right.val) /2
        node.left = None
        node.right = None
        node.col = None
        return node      
    
    def __prune_sub_tree(self, node, x, y) -> TreeNode:
        #首先判断x、y非空。如果测试集为空，将树做塌陷处理。
        #   递归的将左右子树的均值替换为子树的均值。
        #   并设置节点的col为None。
        # 然后将x、y根据决策树要求进行划分。
        # 然后递归判断左右子树是否可以剪枝
        # 当左右子树都叶子节点时，则可以对当前节点进行剪枝。
        #   计算测试集上，左子树的误差减去左子树的均值 的平方 加上 右子树的。 减去 当前节点的测试集
        #   if (l-l_mean)^2 + (r-r_mean)^2 < (x-(l_mean+r_mean)/2)^2
        #       剪枝
        #   else: 返回当前树
        # 返回当前节点。
        
        # 如果当前节点是叶子节点，不需要剪枝。返回
        if node.col is None:
            return node
        # 如果测试集为空，则坍缩当前子树为叶子节点
        if len(x) == 0:
            return self.__mean_sub_tree(node)
        split_index = x[node.col] < node.val
        left_x, left_y = x[split_index], y[split_index]
        right_x, right_y = x[~split_index], y[~split_index]
        node.left = self.__prune_sub_tree(node.left, left_x, left_y)
        node.right = self.__prune_sub_tree(node.right, right_x, right_y)
        
        # 如果左右子树都是叶子节点可以剪枝。
        # 否则不剪枝。
        if node.left.col is None and node.right.col is None:
            # 进行剪枝
            errorNoMerge = (left_y - node.left.val).pow(2).sum() + (right_y - node.right.val).pow(2).sum()
            errorMerge = (y - (node.left.val+node.right.val)/2).pow(2).sum()
            if errorMerge < errorNoMerge:
                return self.__mean_sub_tree(node)
            else:
                return node
        else:
            return node        
        
    
    def __build_tree(self, x:pd.DataFrame, y:pd.DataFrame) -> Dict[str, Union[str,int]]:
        # 如果y里所有的值都相同，不分裂。 
        # choose best feature
        # 对特征循环：
        # 
        #   获取该列特征所有值的排序，获取每两个元素的中位数作为分裂点。
        #   对于每一个分裂点：
        #       将数据划分成两列。左子树小于，右子树大于等于
        #       使用criterion计算两列的误差。
        #       更新最小的误差和划分点。
        # 比较所有特征，获取最佳特征划分特征和最佳的划分点。
        # 如果误差小于min_impurity_decrease，则不分裂，返回leafType作为值。
        # 如果分裂出的叶子节点的数量小于min_samples_leaf，则不分裂，返回leafType作为均值
        # 使用分裂的值将sample分成左右两部分。左小右大。设置TreeNode
        if len(set(y)) == 1:
            return TreeNode(self.leafType(y))
        
        # 选择出最佳的分割特征和分割点
        best_col, best_split_value, loss = self.__choose_best_feature_to_split(x, y)
        sub_index = x[best_col]<best_split_value
        left_x, left_y = x[sub_index], y[sub_index]
        right_x, right_y = x[~sub_index], y[~sub_index]
        
        # 如果分裂出的叶子节点的数量小于min_samples_leaf，则不分裂，返回leafType作为均值
        if len(left_x) < self.min_samples_leaf or len(right_x) < self.min_samples_leaf: 
            return TreeNode(self.leafType(y))
        # 如果误差小于min_impurity_decrease，则不分裂，返回leafType作为值。
        impurity = self.__calc_criterion(y) - loss
        if impurity < self.min_impurity_decrease:
            return TreeNode(self.leafType(y)) # 叶子节点
        
        # 遍历划分子树。
        left_node = self.__build_tree(left_x, left_y)
        right_node = self.__build_tree(right_x, right_y)
        return TreeNode(best_split_value, best_col, left_node, right_node)
            

    def __choose_best_feature_to_split(self, x:pd.DataFrame, y:pd.DataFrame) -> Tuple[str, float]:
        #获取该列特征所有值的排序，获取每两个元素的中位数作为分裂点。
        #   对于每一个分裂点：
        #       将数据划分成两列
        #       使用criterion计算两列的误差。
        #       更新最小的误差和划分点。
        minLoss = np.inf
        best_col = None
        best_split_value = None
        for col in x.columns:
            value_ascend = sorted(list(set(x[col])))
            for i in range(1, len(value_ascend)):
                split_value = (value_ascend[i] + value_ascend[i-1]) /2
                sub_index = x[col]<split_value
                left_sample = y[sub_index]
                right_sample = y[~sub_index]
                loss = self.__calc_criterion(left_sample) + self.__calc_criterion(right_sample)
                if loss < minLoss:
                    minLoss = loss
                    best_col = col
                    best_split_value = split_value
        return best_col, best_split_value, minLoss
                    
    
    def __calc_criterion(self, y:pd.Series) -> float:
        # 使用criterion计算误差。
        if self.criterion == "mse":
            v = y.var()
            v = 0 if np.isnan(v) else v # 当只有一个值的时候，这个值无法计算方差，所以设置其为0.
            return v * len(y)
        
  
        
    def leafType(self, y:pd.Series):
        return y.mean()
    
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 转换为float类型
        # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat
    
if __name__ == '__main__':
    # 基础功能测试
    myData = loadDataSet('01_DicisionTree/data/ex00.txt') # 检测在ex00没有问题。
    myData = np.array(myData)
    x, y = myData[:,0:1], myData[:, 1]
    
    cart = CART(min_impurity_decrease=1, min_samples_leaf=4)
    cart.fit(x, y)
    d = cart.get_decision_tree_dict()
    print(d)
    print("")
    
    # 测试预测
    myData = loadDataSet('01_DicisionTree/data/ex0.txt') # 检测在ex0 不清楚有无问题
    myData = np.array(myData)
    x, y = myData[:,0:2], myData[:, 1]
    
    cart = CART(min_impurity_decrease=1, min_samples_leaf=4)
    cart.fit(x, y)
    d = cart.get_decision_tree_dict()
    print(d)
    x = x[0]
    cart.prediction(x)
    print("")
    
    
    
    # 测试剪枝
    data = loadDataSet('01_DicisionTree/data/ex2.txt')
    data = np.array(data)
    x, y = data[:,0], data[:, 1]
    cart.fit(x, y)
    
    testData = loadDataSet("01_DicisionTree/data/ex2test.txt")
    testData = np.array(testData)
    x, y = testData[:,0], testData[:, 1]
    cart.prune(x, y)
    print("")