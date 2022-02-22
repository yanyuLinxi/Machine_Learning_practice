import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from math import log
import operator
import pickle
"""
函数说明：获取决策树叶子结点的数目

Parameters:
    myTree - 决策树
    
Returns:
    numLeafs - 决策树的叶子结点的数目
    
Modify:
    2018-07-17
""" 
def getNumLeafs(myTree):
    # 初始化叶子
    numLeafs = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,所以不能用
    # myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next(iter(myTree))
    # 获取下一组字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此节点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


"""
函数说明：获取决策树的层数

Parameters:
    myTree - 决策树
    
Returns:
    maxDepth - 决策树的层数
    
Modify:
    2018-07-17
"""   
def getTreeDepth(myTree):
    # 初始化决策树深度
    maxDepth = 0
    # python3中myTree.keys()返回的是dict_keys,不是list,所以不能用
    # myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    # next() 返回迭代器的下一个项目 next(iterator[, default])
    firstStr = next(iter(myTree))
    # 获取下一个字典
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此节点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        # 更新最深层数
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    # 返回决策树的层数
    return maxDepth


"""
函数说明：绘制结点

Parameters:
    nodeTxt - 结点名
    centerPt - 文本位置
    parentPt - 标注的箭头位置
    nodeType - 结点格式
    
Returns:
    None
    
Modify:
    2018-07-17
"""  
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    # 设置中文字体
    font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=14)
    # 绘制结点createPlot.ax1创建绘图区
    # annotate是关于一个数据点的文本
    # nodeTxt为要显示的文本，centerPt为文本的中心点，箭头所在的点，parentPt为指向文本的点
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', 
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, 
                            arrowprops=arrow_args, FontProperties=font)


"""
函数说明：标注有向边属性值

Parameters:
    cntrPt、parentPt - 用于计算标注位置
    txtString - 标注内容
    
Returns:
    None
    
Modify:
    2018-07-17
""" 
def plotMidText(cntrPt, parentPt, txtString):
    # 计算标注位置（箭头起始位置的中点处）
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


"""
函数说明：绘制决策树

Parameters:
    myTree - 决策树（字典）
    parentPt - 标注的内容
    nodeTxt - 结点名
    
Returns:
    None
    
Modify:
    2018-07-17
""" 
def plotTree(myTree, parentPt, nodeTxt):
    # 设置结点格式boxstyle为文本框的类型，sawtooth是锯齿形，fc是边框线粗细
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xoff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yoff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yoff = plotTree.yoff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        # 如果是叶结点，绘制叶结点，并标注有向边属性值
        else:
            plotTree.xoff = plotTree.xoff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xoff, plotTree.yoff), cntrPt, leafNode)
            plotMidText((plotTree.xoff, plotTree.yoff), cntrPt, str(key))
    plotTree.yoff = plotTree.yoff + 1.0 / plotTree.totalD
    

"""
函数说明：创建绘图面板

Parameters:
    inTree - 决策树（字典）
    
Returns:
    None
    
Modify:
    2018-07-17
""" 
def createPlot(inTree):
    # 画出整个决策树的图。
    # 创建fig
    fig = plt.figure(1, facecolor="white")
    # 清空fig
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 获取决策树叶结点数目
    plotTree.totalW = float(getNumLeafs(inTree))
    # 获取决策树层数
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xoff = -0.5 / plotTree.totalW
    plotTree.yoff = 1.0
    # 绘制决策树
    plotTree(inTree, (0.5, 1.0), '')
    # 显示绘制结果
    plt.show()
    
if __name__ == '__main__':
    from id3 import decision_tree, createDataSet
    x, y, label = createDataSet()
    dt = decision_tree()
    dt.fit(x, y)
    createPlot(dt.decision_tree_)