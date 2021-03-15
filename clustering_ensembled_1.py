import pandas as pd
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame, Series
from sklearn.cluster import KMeans, Birch
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import spectral_clustering
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth

# 读取数据
filepath = "KaggleImbalanced.csv"
df = pd.read_csv(filepath)
feats = [x for x in df.columns if x != 'ProtocolName']
X = df[feats]
# X = X[1:100]
Y = df['ProtocolName']
# Y = Y[1:100]
print(Y.unique())
# [3 1 5 4 2 6]

'''
# 用k-means聚类 聚成k类
def kmeans1(data,k):
    iteration = 50
    data_zs = 1.0*(data - data.mean())/data.std()  # 数据标准化
    model = KMeans(n_clusters = k,max_iter = iteration)  # 分为k类，并发数4
    model.fit(data_zs)  # 开始聚类
    #   聚类结果
    r1 = pd.Series(model.labels_).value_counts()  # 统计各个类别的数目
    r2 = pd.DataFrame(model.cluster_centers_)  # 找出聚类中心
    r = pd.concat([r2,r1],axis = 1)  # 横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(data.columns) + [u'类别数目']  # 重命名表头
    #   详细输出原始数据及其类别
    r = pd.concat([data,pd.Series(model.labels_,index = data.index)],axis = 1)
    #   输出每个样本对应的类别
    # r.columns = list(data.columns) + [u'聚类类别']  #重命名表头
    return (r)

C1 = kmeans1(X,13)
print("C1:", C1)
'''

X = X[1:100]
# 层次聚类
def agglomerative(data,group_size):
    cls = AgglomerativeClustering(n_clusters = group_size,linkage = 'ward')
    cluster_group = cls.fit(data)
    # cnames = ['Blue','Red']

    #   聚类结果
    r1 = pd.Series(cluster_group.labels_).value_counts()  # 统计各个类别的数目
    #   详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(cluster_group.labels_, index=data.index)], axis=1)
    return (r)

C2 = agglomerative(X,6)
print("C2:", C2)
# print(C2.iloc[:,-1].unique())

# Birch聚类
# 设置birch函数
birch = Birch(n_clusters = 6)
# 训练数据
C3 = birch.fit_predict(X)
print("C3:", C3)

# 谱聚类
# 变换成矩阵，输入必须是对称矩阵
metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(X)).astype(np.int32)
metrics_metrix += -1 * metrics_metrix.min()
# 设置谱聚类函数
n_clusters_ = 6
C4 = spectral_clustering(metrics_metrix, n_clusters=n_clusters_)
print("C4:", C4)

# 漂移聚类
# 带宽，也就是以某个点为核心时的搜索半径
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=1000)
# 设置均值偏移函数
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
# 训练数据
ms.fit(X)
# 每个点的标签
C5 = ms.labels_
print("C5:", C5)
# 簇中心的点的集合
# cluster_centers = ms.cluster_centers_
# print('cluster_centers:',cluster_centers)
# 总共的标签分类
labels_unique = np.unique(C5)
# 聚簇的个数，即分类的个数
n_clusters_ = len(labels_unique)
print("number of estimated clusters : %d" % n_clusters_)

# 取C2数据框的最后一列
C2 = C2[C2.columns[-1]]
C2 = np.array(C2)
C2 = Series(C2)
# print(C2)
C3 = np.array(C3)
C3 = Series(C3)
# print(type(C3))
# print(C3)
C4 = np.array(C4)
C4 = Series(C4)
# print(type(C4))
C5 = np.array(C5)
C5 = Series(C5)
# print(type(C5))

#   取出数据在不同聚类结果中的类别情况，输出论文中的a矩阵
a_Matrix = pd.concat([C2, C3, C4, C5], axis=1)
a_Matrix = np.array(a_Matrix)
print(a_Matrix)

#   由a矩阵创建c矩阵（协关联矩阵）
c = []
for l in range(4):  # 4 为聚类算法个数
    for i in range(99):  # 99 为样本个数
        for j in range(i,99):
            if (a_Matrix[i,l] == a_Matrix[j,l]):
                co = 1
            else:
                co = 0
            c.append(co)
print(c)
print(len(c))

# 累加求和
def sum1(n):
    sum = 0
    for i in range(n+1):
        sum = sum + i
    return (sum)

d1 = sum1(99)
d2 = 2*d1
d3 = 3*d1
d4 = 4*d1
print(d1)

c1 = c[0:d1]
c2 = c[d1:d2]
c3 = c[d2:d3]
c4 = c[d3:d4]
c5 = np.array([c1, c2, c3, c4])  # c5求出的是每一种聚类方法聚类后每对样本的关系值
print(c5)

#   求c矩阵的上三角矩阵的元素，即求得每对样本之间的关系稳定值
p1 = []
for i in range(int(len(c)/4)):  # 4 为聚类算法个数
    p2 = float((1 / 4) * (c5[0, i] + c5[1, i] + c5[2, i] + c5[3, i]))
    p1.append(p2)
print(p1)

#   将p1转为上三角矩阵，99 为样本个数，可以参数化，用len(X)代替
tri = np.zeros((99, 99))
iu1 = np.triu_indices(99)
tri[iu1] = p1
print(tri)
#   输出c矩阵
for i in range(99):
    for j in range(i,99):
        temp = tri[i][j]
        tri[j][i] = temp
print(tri)