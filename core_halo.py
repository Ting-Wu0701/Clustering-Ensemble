"""
    寻找簇核和簇环
    输入：计算协关联矩阵，即c矩阵
    输出：簇核样本集，簇环样本集
    算法流程：

            利用 stablity_line（） 或 stablity_quadratic（）计算样本的稳定性
            保存每个样本的稳定性
            利用 find_t（）求得阈值ts
            当样本的稳定度大于阈值ts时将其设定为簇核（ts说明他足够稳定了可以当作聚类核）
            当样本的稳定度小于阈值ts时将其设定为簇环（<ts说明它不稳定可以作为聚类环中的样本）
"""
from sklearn.cluster import AgglomerativeClustering

from clustering_ensembled_1 import tri
from Ostu import stablity_line
from Ostu import stablity_quadratic
from Ostu import find_t
import pandas as pd
import numpy as np

cluster_core = []
cluster_halo = []
def find_core_halo(comatrix,ts):
    # 基于线性函数的稳定性
    sl = list(stablity_line(comatrix, ts))
    # 基于二次函数的稳定性
    # sq = list(stablity_quadratic(comatrix, ts))
    for i in range(len(sl)):  # 注意参数，参数可选（sl,sq）
        if sl[i] >= ts:
            cluster_core.append(i)
        else:
            cluster_halo.append(i)
    return cluster_core,cluster_halo
cluster_core, cluster_halo = find_core_halo(tri, 0.94)
print("簇核：", cluster_core)
print("簇环：", cluster_halo)
"""
存在的问题：
        ts如何确定？？？
"""

"""
寻找聚类核的结构
算法流程：
        提取协关联矩阵中的聚类核样本
        利用层次聚类发现聚类核结构
"""
a1 = pd.DataFrame()
a2 = pd.DataFrame()
def discover_core_structure(comatrix,cluster_core):
    # 提取协关联矩阵中的聚类核样本
    # 提取协关联矩阵中的聚类核样本---先提取簇核样本所在的列
    for i in range(len(cluster_core)):
        j = cluster_core[i]
        a1 = pd.concat([a1, pd.DataFrame(comatrix[:, j])], axis=1)
    a1 = np.array(a1)
    # 提取协关联矩阵中的聚类核样本---再提取簇核样本所在的行
    for i in range(len(cluster_core)):
        j = cluster_core[i]
        a2 = pd.concat([a2, pd.DataFrame(a1[j, :]).T], axis=0)
    a2 = np.array(a2)
    # 此时a2矩阵是簇核样本形成的协关联矩阵，矩阵中的每一项代表的是每一对样本之间的相似度
    # 对簇核样本进行层次聚类
    # 重点：利用a2这个相似度矩阵取代了传统层次聚类中的距离矩阵,依据论文所提出的采用单链接的方式
    clustering = AgglomerativeClustering(linkage='single').fit(a2)
    # 此处存在一个问题：AgglomerativeClustering().fit(X)中X为原始矩阵，fit方法自动将其转化为距离矩阵进行聚类
    # 然而此处我不希望用距离矩阵而是要用相似度矩阵来取代距离矩阵，再用这个方法显然不合适
    return clustering

clustering = discover_core_structure(tri, cluster_core)
labels = clustering.labels_
print(labels)

"""
根据聚类核的结构划分小簇
如：聚类核：[x1,x2,x4,x5,x7,x9],其中x1,x4,x9为一类，x2,x5,x7为一类，则划分两个小簇
   簇1：[x1,x4,x9]  簇2：[x2,x5,x7]
"""
def split_clustering(clustering):
    # 聚类数
    n_clusters = clustering.n_clusters_
    # 类别标签
    labels = clustering.labels_




"""
分配簇环中的样本
        输入：簇核样本集 cluster_core
             簇环样本集 cluster_halo
             协关联矩阵 tri
             簇核结构   clustering
             样本总数   n
        
        输出：预结构     pre_structure

算法流程：
        1: 当簇核中的样本个数<总样本数时，分配簇环中的样本，直到簇环中的所有样本都分配到了簇核中，退出循环
        2: 利用公式（15）计算O’（等待被分给簇核的簇环中的样本）
        3: 从矩阵M(协关联矩阵),O(簇核样本集),O’(等待被分给簇核的簇环中的样本)中提取MOO’
        4: 从公式（16）中获得C’* (将选择好的簇环中的样本分给离他最近的簇，即簇核里的小簇) 
        5: O = O U O’即将簇环中的样本分配到簇核
        6: 直到所有样本都分配给簇核
"""

def Assgine_halo_samples(cluster_core,cluster_halo,comatrix,clustering,n,tr):
    while(len(cluster_core)<n):
        wait_assgine = []
        no_wait_assgine = []
        # 当簇核中的样本数目<总样本数时，分配簇环中的样本
        # 计算簇环中的每个样本到簇核中每个样本的相似度
        for i in range(len(cluster_halo)):
            # max记录簇环中样本与簇核中样本的最大相似度
            max = 0
            for j in range(len(cluster_core)):
                # 计算簇环中样本cluster_halo[i]与簇核中样本cluster_core[j]的接近度/相似度
                pr = comatrix[cluster_halo[i],cluster_core[j]]
                if(max > pr):
                    max = max
                else:
                    max = pr
                    # 记录该样本的标签-----------------------
                    label = cluster_core[j].label
            # 当最大相似度>一定阈值后将其合并到预分配簇环样本集中
            if (max > tr):
                wait_assgine.append(cluster_halo[i])
                # 将该待分配的簇环样本分到簇核中
                cluster_core.append(cluster_halo[i])

                # 将该待分配的簇环样本从簇环样本集中移除
                cluster_halo.remove(cluster_halo[i])
            else:
                no_wait_assgine.append(cluster_halo[i])


