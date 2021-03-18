from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.cluster.hierarchy as sch
"""
def fcluster(Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None):
    Z代表了利用“关联函数”关联好的数据       本例使用的是相似度矩阵
    t这个参数是用来区分不同聚类的阈值，在不同的criterion条件下所设置的参数是不同的。       此处t的值应该等于Ostu输出的阈值   
    1. 当criterion为’inconsistent’时，t值应该在0-1之间波动，t越接近1代表两个数据之间的相关性越大，t越趋于0表明两个数据的相关性越小。
    这种相关性可以用来比较两个向量之间的相关性，可用于高维空间的聚类       正适用于我的情况，哈哈
    2. 当criterion为’distance’时，t值代表了绝对的差值，如果小于这个差值，两个数据将会被合并，当大于这个差值，两个数据将会被分开。
    3. 当criterion为’maxclust’时,t代表了最大的聚类的个数，设置4则最大聚类数量为4类，当聚类满足4类的时候，迭代停止
    4. 当criterion为’monocrit’时，t的选择不是固定的，而是根据一个函数monocrit[j]来确定。

# 进行层次聚类:
Z = sch.linkage(sim_matrix, method='single')
Z共有四列组成:
        第一字段与第二字段分别为聚类簇的编号，在初始距离前每个初始值被从0~n-1进行标识，每生成一个新的聚类簇就在此基础上增加一对新的聚类簇进行标识，
        第三个字段表示前两个聚类簇之间的距离，
        第四个字段表示新生成聚类簇所包含的元素的个数

"""

c = pd.DataFrame([[1, 1, 0.33, 0, 0, 0],[1, 1, 0.33, 0, 0, 0],[0.33, 0.33, 1, 0.67, 0.33, 0.33],[0, 0, 0.67, 1, 0.67, 0.67],[0, 0, 0.33, 0.67, 1, 1],[0, 0, 0.33, 0.67, 1, 1]])
c = np.array(c)
print(c)

"""
基于协关联矩阵的层次聚类函数：
    param1:co_matrix:协关联矩阵，即c矩阵
    param2:n:样本总数
    param3:thre:Ostu输出的阈值
"""
def HC(co_matrix,n,thre):
    # 生成全1的矩阵，矩阵大小为样本总数*样本总数
    c1 = np.ones((n, n))
    # 由于c矩阵是协关联矩阵，c[i,j]反应的是第i个样本和第j个样本的关系稳定性（相似度），越接近于1越应该聚在一起
    # 然而传入到层次聚类时，层次聚类会将其作为距离矩阵，越近于1越代表两个点之间的距离越大
    # 因此，我们用全1矩阵减去协关联矩阵，相当于将样本间的稳定性转换为距离，相似度高的距离就小了
    # 相似度矩阵
    sim_matrix = c1 - co_matrix
    # 进行层次聚类:
    Z = sch.linkage(sim_matrix, method='single')
    # fig = plt.figure(figsize=(5, 3))
    # dn = dendrogram(Z)
    # # 保存生成的层次聚类树
    # plt.savefig('plot_dendrogram.png')
    # plt.show()
    # print(Z)
    # 此处thre的值应该等于Ostu输出的阈值
    cluster = sch.fcluster(Z, t=thre, criterion='inconsistent')
    return cluster
result = HC(c, 6, 0.66)
print("基于样本稳定性的层次聚类结果：",result)

# 此时输出结果：[1 1 4 3 2 2]，结果显示分成了4类，第1个第2个样本属于类1，第5个第6个样本属于类2，第4个样本属于类3，第3个样本属于类4
# 由图可以看出和聚类树结果相同

"""
寻找核结构：
    input：core：聚类核样本
           co_matrix：协关联矩阵
           thre：阈值，即Ostu输出的阈值
    输出：核结构
"""
# 假设以第1、2、3、4、5个样本作为核样本
core = [0, 1, 2, 3, 4]

def discovering_Core_Structure(core,co_matrix,thre):
    # core_co_matrix 为核样本的协关联矩阵
    core_co_matrix = np.zeros((len(core), len(core)))
    # 提取核样本的协关联矩阵
    for i in range(len(core)):
        for j in range(len(core)):
            core_co_matrix[i, j] = co_matrix[core[i], core[j]]
    core_structure = HC(core_co_matrix, len(core), thre)
    return core_structure
core_structure = discovering_Core_Structure(core,c,0.66)
print("簇核结构：",core_structure)
# 此时输出结果为[1 1 3 2 2]
# 正确结果应该是[1 1 3 4 2]
# 可以看出结果比较接近，产生偏差的原因应该是样本太少了的缘故，只有5个样本难免会有所偏差

# 现在有一个比较重要的问题是，层次聚类无法人为规定聚类数，若想聚成自己希望的聚类数，只能通过绘制聚类树和输出Z，人工查找


# 假设簇核集样本为第1、3、4、5个样本
core = [0, 2, 3, 4]
# 假设簇环集样本为第2、6个样本
halo = [1, 5]
# 如果能将halo中的样本的类别标记为[1,2]则说明标记对了
# 假设核结构为【1 3 4 2】
core_structure = [1, 3, 4, 2]
"""
分配簇环中的样本
    input：core：簇核样本集
           halo：簇环样本集
           co_matrix：协关联矩阵
           co_structure：核结构
    output：pre_structure：预结构，即将簇环中的样本分配给簇核中的小簇
"""
def assgine_Halo_Samples(core,halo,co_matrix,core_structure):
    # n_clusters 为簇核中小簇的数目，即把簇核集分成了几类
    n_clusters = len(np.unique(core_structure))
    # 分配簇环中的样本
    for j in range(len(halo)):
        # max_sim_value最大相似度的值
        max_sim_value = 0
        for i in range(len(core)):
            # s为簇环样本halo[j]与簇核样本core[i]的稳定性，稳定性越接近于1代表相似度越高
            s = co_matrix[halo[j], core[i]]
            if s > max_sim_value:
                max_sim_value = s
                # 与该簇环样本相似度最高的簇核样本的类别号为该簇环样本的类别号
                max_sim_index = core_structure[i]
            # else:
            #     max_sim_value = 0
            #     max_sim_index = None
        # 将该簇环样本加入到簇核样本中
        core.append(halo[j])
        # 标记该簇环样本的类别号
        core_structure.append(max_sim_index)
    pre_structure = core_structure
    return pre_structure
pre_structure = assgine_Halo_Samples(core,halo,c, core_structure)
print("预结构：",pre_structure)
# 输出结果为[1, 3, 4, 2, 1, 2]
# 前四个为簇核样本的类别号，后两个为簇环样本的标记号，标记对了，哦耶