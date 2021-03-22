"""
当我们获取的预结构的聚类数大于实际的聚类数时，我们需要对预结构中的簇进行调整（合并）
此处我们选用k-means算法进行簇与簇之间的合并

"""
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

# 假设簇核集样本为第1、3、4、5个样本
core = [0, 2, 3, 4]
# 假设簇环集样本为第2、6个样本
halo = [1, 5]
# 如果能将halo中的样本的类别标记为[1,2]则说明标记对了
# 假设核结构为【1 3 4 2】
core_structure = [1, 3, 4, 2]
pre_structure = [1, 3, 4, 2, 1, 2]
# 组合核-环-预结构
core1_halo1 = core + halo
print(core1_halo1)
core1_halo1 = pd.DataFrame(core1_halo1)
pre_structure1 = pd.DataFrame(pre_structure)
table = pd.concat((core1_halo1,pre_structure1),axis=1)
table.columns = list(['num_sample','label'])
print(table)

x1 = [100,100]
x2 = [95,95]
x3 = [70,70]
x4 = [50,53]
x5 = [30,30]
x6 = [20,20]

X = np.vstack((x1, x3, x4, x5, x2, x6))
# table1为组合好的矩阵，第一列代表样本编号，第二列代表样本编号对应的类别标签，其余列为样本编号对应的样本数据
table1 = np.hstack((table,X))
# table2为去除掉样本编号的矩阵，第1列代表类别标签，其余列为样本数据
table2 = table1[:, 1:table1.shape[1]]

table2 = pd.DataFrame(table2)
print(table2)
combine = table2.groupby(table2.loc[:,0])
# print(combine.mean())
combine1 = np.array(combine.mean())
print(combine1)

combine2 = pd.DataFrame(combine1)
# 用k-means聚类 聚成k类
def kmeans1(data,k):
    iteration = 50
    data_zs = 1.0*(data - data.mean())/data.std()  # 数据标准化
    model = KMeans(n_clusters=k, max_iter=iteration)  # 分为k类，并发数4
    model.fit(data_zs)  # 开始聚类
    #   详细输出原始数据及其类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    return (r)

result = kmeans1(combine2,2)
print(result)

table2 = np.array(table2)
for i in range(0,table2.shape[0]):
    if table2[i, 0] == 1 or table2[i, 0] == 3:
        table2[i, 0] = 0
    else:
        table2[i, 0] = 1
print(table2)
