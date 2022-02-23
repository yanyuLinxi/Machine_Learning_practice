
# 集成学习

集成学习通过叠加弱分类器，来训练得到一个强分类器。
根据叠加的方式不同，可以分为Bagging, Boosting, Stacking.

Bagging: 多次采样，并行训练，集体投票，减少方差
Boosting：层层叠加，串行训练，聚焦错误，减少偏差
Stacking：多次采样，并串结合，输出作为最后的输入


# Adaboost

Adaboost 使用加法模型，将各个弱分类器组成强分类器。

特点:
1. 采用指数函数$y=e^{-f(x)h(x)}$作为损失函数

步骤：
1. 初始化第一个分类器的权重
2. 根据前一个分类器的训练结果，依次调整后面样本的权重。
3. 将弱分类器组成一个强分类器。误差率小的弱分类器占据的权重更大。

具体步骤：
1. 初始化权重
2. 进行t=1...T轮迭代
   1. 选取当前误差最低的弱分类器h作为第t个分类器。使用该分类器去拟合带权重的标签。将带权重的标签作为下一轮拟合的目标。
   2. 计算弱分类器在分布D_t的分类误差e_t。
   3. 计算误差率。$\alpha_t = \frac{1}{2}ln\frac{1-e_t}{e_t}$
   4. 更新样本的权值分布。
3. 最终按照误差率$\alpha_t$来组合所有的弱分类器。

Notes：

1. 样本权重在决策树计算的时候。和误差乘起来，即增大错误分类的误差权重，减少正确分类的误差权重。通过这种方式做到决策树去拟合误差。

公式推导：
1. 计算为什么可以使用指数函数
2. 计算基分类器的权重
3. 计算下一轮数据的分布。且分析为什么这样设置样本权重。


# 问题
1. 为什么使用指数函数作为损失函数



2. adaboost优缺点

优点：可以得到较好的效果
缺点：1. 对异常样本敏感（异常样本会不停的增加权重，影响效果。）。2. 只能采用指数损失函数，做二分类学习任务。