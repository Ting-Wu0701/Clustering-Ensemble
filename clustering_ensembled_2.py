import Parameters as Para
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.cluster import KMeans, Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import spectral_clustering
from sklearn import metrics
from sklearn.cluster import MeanShift, estimate_bandwidth
# 读取数据

df = pd.read_csv(Para.filepath)
feats = [x for x in df.columns if x != 'ProtocolName']
X = df[feats]
X = X[1:100]
Y = df['ProtocolName']
Y = Y[1:100]
print(Y.unique())

# 用k-means聚类 聚成k类
# def kmeans1(data,k):
#     iteration = 50
#     data_zs = 1.0*(data - data.mean())/data.std()  # 数据标准化
#     model = KMeans(n_clusters=k, max_iter=iteration)  # 分为k类，并发数4
#     model.fit(data_zs)  # 开始聚类
#     #   详细输出原始数据及其类别
#     r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
#     return r
# C1 = kmeans1(X,6)
# print(C1)

# 层次聚类
def agglomerative(data,group_size):
    cls = AgglomerativeClustering(n_clusters=group_size, linkage='ward')
    cluster_group = cls.fit(data)
    #   详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(cluster_group.labels_, index=data.index)], axis=1)
    return r

C2 = agglomerative(X,6)
print("C2:", C2)

# Birch聚类
def birch(data, k):
    birch = Birch(n_clusters=k)
    r = birch.fit_predict(data)
    return r
C3 = birch(X,6)
print("C3:", C3)

# 谱聚类
def spectral(data,k):
    # 变换成矩阵，输入必须是对称矩阵
    metrics_metrix = (-1 * metrics.pairwise.pairwise_distances(data)).astype(np.int32)
    metrics_metrix += -1 * metrics_metrix.min()
    # 设置谱聚类函数
    n_clusters_ = k
    r = spectral_clustering(metrics_metrix, n_clusters=n_clusters_)
    return r
C4 = spectral(X,6)
print("C4:", C4)

# 漂移聚类
def mean_Shift(data):
    # 带宽，也就是以某个点为核心时的搜索半径
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=1000)
    # 设置均值偏移函数
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # 训练数据
    ms.fit(data)
    # 每个点的标签
    r = ms.labels_
    return r
C5 = mean_Shift(X)
print("C5:", C5)


# 取C2数据框的最后一列
C2 = C2[C2.columns[-1]]
C2 = np.array(C2)
C2 = Series(C2)

C3 = np.array(C3)
C3 = Series(C3)

C4 = np.array(C4)
C4 = Series(C4)

C5 = np.array(C5)
C5 = Series(C5)

#   取出数据在不同聚类结果中的类别情况，输出论文中的a矩阵
a_Matrix = pd.concat([C2, C3, C4, C5], axis=1)
a_Matrix = np.array(a_Matrix)
print(a_Matrix)

# 构造协关联矩阵
co_matrix = np.ones((len(X),len(X)))  # len(X)为样本个数

for m in range(len(X)-1):  # m为第m个样本
    for n in range(m+1, len(X)):  # n为第n个样本
        sum_support = 0
        for l in range(4):  # 3为参数，代表算法个数
            if a_Matrix[m, l] == a_Matrix[n, l]:  # 如果第l个算法认为第m个样本和第n个样本是同一类
                sum_support += 1  # 支持在一起的票数+1
            else:  # 如果第l个算法都没有选中fm,fn
                sum_support += 0  # 支持在一起的票数+0
        co_matrix[m,n] = sum_support/4  # 计算第m个样本和第n个样本之间的稳定性 = 支持在一起的票数（即支持在一起算法数）/总算法数
        # 以上得到的co_matrix仅是上三角矩阵，将其进行对称矩阵变换
        co_matrix[n, m] = co_matrix[m, n]
print(co_matrix)

# 构造b矩阵
# get_dummies 是利用pandas实现one hot encode的方式
b = pd.DataFrame()
for i in range(a_Matrix.shape[0]):
    int_as_str = [str(number) for number in a_Matrix[i, :]]
    int_as_str = pd.DataFrame(int_as_str).T
    b = pd.concat((b,int_as_str),axis=0)
print(b)
# b.columns = ['first',  'second', 'third']
b_Matrix = pd.get_dummies(b)
print(b_Matrix)

# 比较两个数组相同位置相同值有几个
def campare(arr1,arr2,val):
    sum = 0
    for i in range(len(arr1)):
        if arr1[i] == arr2[i] and arr1[i] == val:
            sum +=1
    return sum

# 构造d矩阵
d = np.array(b_Matrix)
print(d)
aaa = []
for i in range(d.shape[1]):
    c1 = d[:, i]
    for l in range(d.shape[1]):
        c2 = d[:, l]
        # num为b矩阵中第i列与第j列相同1的个数，反应的是不同算法在聚类时重合的样本数量
        num = campare(c1, c2, 1)
        aaa.append(num)
aaa = np.array(aaa)
d_matrix = aaa.reshape(d.shape[1],d.shape[1])
print(d_matrix)
