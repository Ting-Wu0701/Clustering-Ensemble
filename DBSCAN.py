from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle  ##python自带的迭代器模块
from sklearn.preprocessing import StandardScaler
import pandas as pd
"""
该算法目前效果十分不好
DBSCAN（Density-Based Spatial Clustering of Applications with Noise,具有噪声的基于密度的聚类方法）是一种基于密度的空间聚类算法。
该算法将具有足够密度的区域划分为簇(即要求聚类空间中的一定区域内所包含对象的数目不小于某一给定阈值)，并在具有噪声的空间数据库中发现任意形状的簇，
它将簇定义为密度相连的点的最大集合。
    a）sklearn.cluster.DBSCAN

    b）主要参数（详细参数）

    eps:两个样本之间的最大距离，即扫描半径

    min_samples ：作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身)。

    c）主要属性

    core_sample_indices_:核心样本指数。

    labels_:数据集中每个点的集合标签给,噪声点标签为-1。

"""

filepath = "KaggleImbalanced.csv"
df = pd.read_csv(filepath)
feats = [x for x in df.columns if x != 'ProtocolName']
X = df[feats]
X = X[1:5000]
Y = df['ProtocolName']

X = StandardScaler().fit_transform(X)


db = DBSCAN(eps=0.8, min_samples=20).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('聚类个数: %d' % n_clusters_)
clus = dict();
for label in range(n_clusters_):
    print('Cluster ', label, ':')
    print(list(X[labels == label].flatten()))

"""
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=14)
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
plt.title('the number of clusters: %d' % n_clusters_)
# plt.show()
"""