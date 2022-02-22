from typing import List, Union, Dict
import pandas as pd
from collections import Counter, defaultdict
from math import log
from enum import Enum

class CRITERION(Enum):
    Entropy = 1
    Entropy_rate = 2
    
class decision_tree:
    def __init__(self, criterion:str="id3"):
        # 我们定义一颗决策树。并和sklearn的函数保持适配。实现fit, predict两个通用api
        # 基本思路：
        # fit中传入数据，建立决策树
        #   数据格式dataset: List[Union[int,str]]
        #   调用子函数，建立树。传入dataset，传入行，传入列。
        #       结束条件：所有的行的label是一样的。或者列属性集没有了。
        #       对传入的列，选择增益率最大的一个特征
        #       获取筛选好的列编号。
        #       然后根据特征的值进行划分行。获取行编号。
        #       根据行编号、列编号建立判断字典规则。
        
        # 不一定所有的数据都有列名。很多数据没有列名。所以不应该通过列名来访问。但是后面可视化的时候有列名又比较好。
        # 先想想怎么划分？
        # 我们要不要使用pandas？使用pandas吧。pandas用起来方便。然后传进来的数据不是pandas可以改成pandas
        # 通过一个字典建立判断规则么？有没有更好的方式？即使是连续值也可以通过字典来判断。当字典的第一个key等于第二个key的时候，证明第一个是小于，第二个是大于
        if criterion.lower() in ["id3"]:
            self.criterion = CRITERION.Entropy # entropy, entropy_rate, gini
        elif criterion.lower() in ["c4.5"]:
            self.criterion = CRITERION.Entropy_rate
    
    def fit(self, x:Union[pd.DataFrame,List], y) -> None:
        # 传入的数据可能是list或者pandas的结构
        try:
            if type(x) == list:
                x = pd.DataFrame(x)
            if type(y) not in [pd.DataFrame, pd.Series]:
                y = pd.Series(y)
            assert type(x) == pd.DataFrame, "dataset的数据类型应该为pandas.DataFrame或者可以转换成DataFrame"
        except Exception:
            raise Exception #不解决异常，直接抛出。
        
        assert len(x) == len(y), "数据的长度和标签长度不一致"
        
        self.decision_tree_ = self.__build_tree(x, y)
        print("")
        
        
    def predict(self, x:Union[pd.DataFrame,List]) -> str:
        try:
            if type(x) == list:
                x = pd.DataFrame(x)
            assert type(x) == pd.DataFrame, "dataset的数据类型应该为pandas.DataFrame或者可以转换成DataFrame"
        except Exception:
            raise Exception #不解决异常，直接抛出。
        res = [self.__predict_one(x.iloc[i]) for i in range(len(x))]
        return res
            
    def __predict_one(self, x:pd.Series) -> str:
        res = self.decision_tree_
        while type(res) == dict:
            key = list(res.keys())[0]
            x_v = x[key]
            res = res[key][x_v]
        return res
        
    
    def __build_tree(self, x:pd.DataFrame, y:Union[pd.DataFrame,pd.Series]) -> Dict[str, Union[str,int]]:
        # 返回值Dict中的第一个是列名，第二个是这个列对应的值，该值可能是int或者str
        
        # 结束条件：所有的行的label是一样的。或者列属性集没有了。
        #       对传入的列，选择增益率最大的一个特征
        #       获取筛选好的列编号。
        #       然后根据特征的值进行划分行。获取行编号。
        #       根据行编号、列编号建立判断字典规则。
        row_columns = x.columns
        label_set = set(y)
        if len(label_set) == 1: # 所有label统一了
            return list(label_set)[0]
        # 然后开始生成这一棵树。
        info_gain_col = self.__choose_best_feature(x, y) # 根据信息增益或者信息增益率选取最优的特征进行分裂。
        sub_cols = [col for col in row_columns if col != info_gain_col] # 获取子列。
        col_value_tree = dict() # =Dict[col_value: sub_tree] # 建立子判断树。
        if len(sub_cols) == 0:
            for value in set(x[info_gain_col]): # 将样本集进行划分
                sub_index = x[info_gain_col]==value
                col_value_tree[value] = y[sub_index].value_counts().index[0]  # 终止条件。当所有的列都遍历完时。
        else:
            for value in set(x[info_gain_col]):
                sub_index = x[info_gain_col]==value
                col_value_tree[value] = self.__build_tree(x.loc[sub_index, sub_cols], y[sub_index]) 
        sub_tree = {info_gain_col:col_value_tree} # =Dict[col_name:col_value_tree]
        return sub_tree

    def __choose_best_feature(self, x, y):
        if self.criterion == CRITERION.Entropy:
            info_gain_max = -1
            info_gain_col = None
            for col in x.columns:
                info_gain = self.__calc_feature_info_gain(x[col], y) # 计算每个特征的信息增益。并选择信息增益最大的一个。
                if info_gain > info_gain_max:
                    info_gain_max = info_gain
                    info_gain_col = col
            return info_gain_col
        elif self.criterion == CRITERION.Entropy_rate:
            info_gains = pd.DataFrame([self.__calc_feature_info_gain(x[col], y, calc_intrinsic=True) for col in x.columns], index=x.columns, columns=["gain", "int"]) # 获取每一列的信息增益和固有值。
            info_gains = info_gains[info_gains["gain"]>=info_gains["gain"].mean()] # 筛选出信息增益大于平均值的列
            info_gains["gain_rate"] = info_gains["gain"] / info_gains["int"] # 信息增益率=信息增益/固有值。
            return info_gains["gain_rate"].idxmax(0) # 返回增益率最大的列。
        else:
            raise "Unkown Criterion 未知的决策树判别标准"

    def __calc_shannon_entropy_labels(self, labels: List[str]) -> float:
        # 仅仅根据标签来计算香农熵。
        # 根据label计算香农熵。
        label_count = Counter()
        data_nums = len(labels)
        for label in labels:
            label_count[label] += 1
        for key in label_count:
            label_count[key] = label_count[key]/data_nums # = p_k
            label_count[key] = label_count[key]*log(label_count[key], 2) # =p_k*(log2(p_k))
        return -sum(list(label_count.values())) # = -\sum


    def __calc_feature_info_gain(self, features:pd.Series, label:pd.Series, calc_intrinsic:bool=False) -> float:
        # 计算dataset第index个特征的信息增益。
        # 对每个特征计算信息增益。
        #   统计每个特征的所有分类。
        #   统计属于不同分类的样本的index
        #   统计不同分类的信息熵。
        #   统计不同分类的信息熵乘以不同分类的个数所占比例
        #   计算信息增益
        sample_classify = defaultdict(lambda:list())
        data_nums = len(features)
        for i in range(len(features)):
            sample_classify[features.iloc[i]].append(label.iloc[i])
        weight = {key:len(sample_classify[key])/data_nums for key in sample_classify} # 每一类分类的权重 D_v/D
        for key in sample_classify: # 计算信息增益 公式 D_v/D * Ent(D_v)
            sample_classify[key] = weight[key] * self.__calc_shannon_entropy_labels(sample_classify[key])
        
        if calc_intrinsic:
            intrinsic = -sum([weight[key]*log(weight[key], 2) for key in weight]) # \sum D_v/D * log(D_v/D) # 计算信息增益率中的固有值
            return -sum(sample_classify.values()), intrinsic
        else:
            return -sum(sample_classify.values()) # 信息增益求和

def createDataSet():
    # machine in action 自创数据集
    # 数据集
    dataSet = [[0, 0, 0, 0, 'no'],
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    # 分类属性
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # 返回数据集和分类属性
    return [x[:-1] for x in dataSet], [x[-1] for x in dataSet], labels

def load_data():
    with open('01_DicisionTree/data/lenses.txt') as fr:
        # 处理文件，去掉每行两头的空白符，以\t分隔每个数据
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    # 提取每组数据的类别，保存在列表里
    lenses_targt = []
    for each in lenses:
        # 存储Label到lenses_targt中
        lenses_targt.append(str(each[-1]))
    # 特征标签
    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    # 保存lenses数据的临时列表
    lenses_list = []
    # 保存lenses数据的字典，用于生成pandas
    lenses_dict = {}
    # 提取信息，生成字典
    for each_label in lensesLabels:
        for each in lenses:
            # index方法用于从列表中找出某个值第一个匹配项的索引位置
            lenses_list.append(each[lensesLabels.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    # 打印字典信息
    # print(lenses_dict)
    # 生成pandas.DataFrame用于对象的创建
    lenses_pd = pd.DataFrame(lenses_dict)
    return lenses_pd, lenses_targt

if __name__ == '__main__':
    """
    # 测试一
    x, y, labels = createDataSet()
    dt = decision_tree()
    dt.fit(x, y)
    p = dt.predict(x)
    print("")
    """
    
    # 测试二
    x, y = load_data()
    dt = decision_tree(criterion="c4.5")
    dt.fit(x, y)
    from plotTree import createPlot
    createPlot(dt.decision_tree_)
    
    