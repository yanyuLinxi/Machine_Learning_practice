# Machine Learning in practice

对机器学习算法的一些实现。

大多算法实现了scikit-learn的接口。可以使用类似的api调用方式进行训练。

大多数算法的传入参数可以使用pandas.dataFrame传入。如果为有监督学习，则标签列名应为"label"。

| 文件夹                                            | 内容                      |
| :------------------------------------------------ | :------------------------ |
| [01_DicisionTree](./01_DicisionTree/)             | id3, c4.5, cart介绍和实现 |
| [02_KNN](./02_KNN/)                               | KNN介绍和实现             |
| [03_AdaBoost](./03_AdaBoost/)                     | Adaboost介绍、公式推导和实现             |
| [04_KMeans](./04_KMeans/)                         | KMeans和Bi-KMeans介绍和实现 ，其他Kmeans的改进版。            |
| [05_LogisticRegression](./05_LogisticRegression/) | 逻辑回归公式推导、实现。包括极大似然概率、交叉熵损失函数的推导。             |


## Notes
1. 文件夹编号仅和实现顺序有关

## Running

每个模型后面都有简短的测试，或有专门的测试文件。

运行示例为: ```python 01_Dicision/cart.py```

## Requirements

### 部分文件需要 

1. numpy>=1.19.5
2. scikit-learn>=0.24.2
3. matplotlib>=3.3.3

## References
1. 书籍
   1. 《机器学习》- 周志华 西瓜书
   2. *Machine learning in Action* - Peter Harrington.
   3. *Hands-on Machine Learning* - Aurelien Geron.

