import heapq
import math
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import preprocessing
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import time
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2

# 读取数据
filepath = "KaggleImbalanced.csv"
df = pd.read_csv(filepath)
df = df.head(3000)
feats = [x for x in df.columns if (x != 'ProtocolName' and x != 'L7Protocol')]
X = df[feats].astype(float)
Y = df['ProtocolName']
# print("类别数：", Y.unique())
data = pd.concat((X, Y), axis=1)

"""
    利用随机森林分类器进行特征选择的函数
    输入：X：样本的所有特征
         Y：样本类别
         k: 特征选择的数量
    输出：k个最重要的特征属性
"""


def RF_feature_importance(X, Y, k):
    # get_dummies 是利用pandas实现one hot encode的方式
    X_one_hot = pd.get_dummies(X, prefix_sep='_')
    # preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    # StandardScaler()标准化数据
    X2 = StandardScaler().fit_transform(X_one_hot)

    # 划分数据集
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X2, Y, test_size=0.3, random_state=101)
    # 利用随机森林分类器进行分类
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train, Y_Train)
    # 获取各特征的重要性
    feat_importance = pd.Series(trainedforest.feature_importances_).tolist()
    for i in range(0, len(feat_importance)):
        if math.isnan(feat_importance[i]):
            feat_importance[i] = -555

    re1 = map(feat_importance.index, heapq.nlargest(k, feat_importance))
    feat_index = list(re1)
    return feat_index

# feature_index = RF_feature_importance(X,Y,6)
# C1 = np.array(feature_index)
# C1 = Series(C1)
# print("fearure_index1：",feature_index)

"""
    利用协相关矩阵进行特征选择的函数
    输入：data：由特征值和类别标签组成的数组
         k：想要获得的特征数+1（或是与类别相关的特征的阈值）
         因为协关联矩阵会自动算上类别标签自己的协关联，因此最重要的特征一定会显示是类别那一列，所以k比我们想要的特征多一个
    输出：筛选出的特征
"""
def c_matrix(data,k):
    # 显示最后一列（类别标签列）与特征的相关系数
    corr_ProtocolName = data.corr()[data.columns[-1]].tolist()

    for i in range(0, len(corr_ProtocolName)):
        if math.isnan(corr_ProtocolName[i]):
            corr_ProtocolName[i] = -555

    re1 = map(corr_ProtocolName.index, heapq.nlargest(k, corr_ProtocolName))
    re1 = list(re1)
    re1 = re1[1:len(re1)]

    # # 只考虑与输出变量至少 0.3 相关的特性
    # highest_corr = corr_ProtocolName[corr_ProtocolName > 0.3]
    # highest_corr.sort_values(ascending=True)

    return re1

# feature_index = c_matrix(data,7)
# C2 = np.array(feature_index)
# C2 = Series(C2)
# print("feature_index2：",feature_index)

"""
    利用递归特征消除法（RFE）随机森林分类器进行特征选择的函数
    输入：X：样本的所有特征
         Y：样本类别
         k: 选取的特征数
    输出：k个最重要的特征属性
"""

def RFE_feature_importance(X,Y,k):
    # get_dummies 是利用pandas实现one hot encode的方式
    X = pd.get_dummies(X, prefix_sep='_')
    # sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内
    # fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
    # 然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
    X2 = StandardScaler().fit_transform(X)

    # n_estimators：在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。
    model = RandomForestClassifier(n_estimators=700)
    rfe = RFE(model, n_features_to_select=k)
    rfe = rfe.fit(X2, Y)
    feat_choose = rfe.support_
    fea_index = []
    for i in range(0, len(feat_choose)):
        if feat_choose[i] == True:
            fea_index.append(i)
        else:
            continue
    return fea_index

# feature_index = RFE_feature_importance(X,Y,6)
# C3 = np.array(feature_index)
# C3 = Series(C3)
# print("feature_index3：",feature_index)



"""
    利用SelectFromModel和随机森林分类器进行特征选择的函数
    输入：X：样本的所有特征
         Y：样本类别
         SelectFromMode无法指定具体的特征数目
    输出：k个最重要的特征属性
"""
def Select_From_Model(X,Y):
    # get_dummies 是利用pandas实现one hot encode的方式
    X = pd.get_dummies(X, prefix_sep='_')
    # sklearn.preprocessing.LabelEncoder()：标准化标签，将标签值统一转换成range(标签值个数-1)范围内
    # fit_transform(trainData)对部分数据先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），
    # 然后对该trainData进行转换transform，从而实现数据的标准化、归一化等等
    Y = preprocessing.LabelEncoder().fit_transform(Y)

    # 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值而主导
    X2 = StandardScaler().fit_transform(X)
    # 模型使用极端随机树
    # model = ExtraTreesClassifier()
    # 模型使用随机森林
    model = RandomForestClassifier(n_estimators=700)
    model = model.fit(X2, Y)
    # SelectFromModel中模型放进去的时候，万一你在放进去之前已经fit完了，总不至于再fit一遍，然后选出来吧。
    # 所以这个时候用到了后面的prefit这个参数。它默认为False，也就是说默认为你还没有fit过，所以可以边fit边select features。
    # 如果已经fit过，直接写prefit=True就可以了直接挑选features，节省时间。
    model = SelectFromModel(model, prefit=True)

    feat_choose = model.get_support()
    fea_index = []
    for i in range(0, len(feat_choose)):
        if feat_choose[i] == True:
            fea_index.append(i)
        else:
            continue
    return fea_index

# feature_index = Select_From_Model(X,Y)
# C4 = np.array(feature_index)
# C4 = Series(C4)
# print("feature_index4：",feature_index)


"""
    利用Select_K_Best和随机森林分类器进行特征选择的函数
    输入：X：样本的所有特征
         Y：样本类别
         k: 特征选择的数量
    输出：k个最重要的特征属性
"""
def Select_K_Best(X,Y,k):
    X = pd.get_dummies(X, prefix_sep='_')
    Y = LabelEncoder().fit_transform(Y)
    X2 = StandardScaler().fit_transform(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    Scaled_X = min_max_scaler.fit_transform(X2)
    # 使用卡方检验，特征数为4
    selector = SelectKBest(chi2, k)
    selector.fit(Scaled_X, Y)
    fearure_index = selector.get_support(True)
    return fearure_index

# feature_index = Select_K_Best(X,Y,6)
# C5 = np.array(feature_index)
# C5 = Series(C5)
# print("feature_index5：",feature_index)



C1 = [3, 1, 2, 0, 22, 5]
C1 = Series(C1)
C2 = [1, 0, 45, 17, 48, 14]
C2 = Series(C2)
C3 = [0, 1, 3, 5, 37, 58]
C3 = Series(C3)
C4 = [0, 1, 2, 3, 5, 7, 12, 13, 19, 20, 22, 24, 25, 27, 36, 37, 38, 50, 55, 57, 58]
C4 = Series(C4)
C5 = [0, 2, 3, 45, 46, 47]
C5 = Series(C5)


#   取出不同算法取出的特征情况，输出论文中的a矩阵
a_Matrix = pd.concat([C1, C2, C3, C4, C5], axis=1)
a_Matrix = np.array(a_Matrix)
print(a_Matrix)

#  将a_Matrix转为更标准的投票矩阵
m1 = np.zeros((60, 5))
for i in range(0, len(C1)):
    m1[int(C1[i]), 0] = 1

for i in range(0, len(C2)):
    m1[int(C2[i]), 1] = 1

for i in range(0, len(C3)):
    m1[int(C3[i]), 2] = 1

for i in range(0, len(C4)):
    m1[int(C4[i]), 3] = 1

for i in range(0, len(C5)):
    m1[int(C5[i]), 4] = 1

print(m1)


# 为a矩阵添加投票数这一列
votes = []
for i in range(60):  # 60为特征个数
    sum_votes = 0
    for j in range(5):  # 5为算法个数
        if m1[i,j] == 1:
            sum_votes += 1
    votes.append(sum_votes)

vote_matrix = np.insert(m1, 5, values=votes, axis=1)  # 在第6列添加votes值
print(vote_matrix)


# 构造正-协关联矩阵
pos_matrix = np.ones((60,60))

for m in range(59):  # m为第m个特征
    for n in range(m+1, 60):  # n为第n个特征
        sum_support = 0  # 支持fm和fn在一起的投票数
        vote_y = 0  # 有效票数，即发表意见的票数
        for l in range(5):  # 5为参数，代表算法个数
            if m1[m, l] == 1 and m1[n, l] == 1:  # 如果第l个算法同时选中了fm,fn
                vote_y += 1  # 有效票数+1
                sum_support += 1  # 支持在一起的票数+1
            elif m1[m, l] == 1 and m1[n, l] == 0:  # 如果第l个算法选中了fm，没有选中fn
                vote_y += 1  # 有效票数+1
                sum_support += 0  # 支持在一起的票数+0
            elif m1[m, l] == 0 and m1[n, l] == 1:
                vote_y += 1  # 有效票数+1
                sum_support += 0  # 支持在一起的票数+0
            else:  # 如果第l个算法都没有选中fm,fn
                vote_y += 0  # 有效票数+0
                sum_support += 0  # 支持在一起的票数+0
        rate = vote_y / 5  # 有效票比例 = （投票的算法数）/总算法数
        if vote_y == 0:  # 有效票数存在为0的可能性
            pos_matrix[m,n] = 0
        else:
            pos_matrix[m, n] = rate * (sum_support / vote_y)  # <fi,fj> = 有效票比例 *（支持在一起的算法数/发表意见的算法数） = 支持在一起的算法数/总算法数

# print(pos_matrix)

# 将正-协关联矩阵变成对称矩阵
for m in range(59):
    for n in range(m+1, 60):
        pos_matrix[n, m] = pos_matrix[m,n]
print(pos_matrix)



# 构造负-协关联矩阵
nag_matrix = np.zeros((60,60))

for m in range(59):  # m为第m个特征
    for n in range(m+1, 60):  # n为第n个特征
        sum_oppose = 0  # 反对fm和fn在一起的投票数
        vote_y = 0  # 有效票数，即发表意见的票数
        for l in range(5):  # 5为参数，代表算法个数
            if m1[m, l] == 1 and m1[n, l] == 1:  # 如果第l个算法同时选中了fm,fn
                vote_y += 1  # 有效票数+1
                sum_oppose += 0  # 反对在一起的票数+0
            elif m1[m, l] == 1 and m1[n, l] == 0:  # 如果第l个算法选中了fm，没有选中fn
                vote_y += 1  # 有效票数+1
                sum_oppose += 1  # 反对在一起的票数+1
            elif m1[m, l] == 0 and m1[n, l] == 1:
                vote_y += 1  # 有效票数+1
                sum_oppose += 1  # 反对在一起的票数+1
            else:  # 如果第l个算法都没有选中fm,fn
                vote_y += 0  # 有效票数+0
                sum_oppose += 0  # 反对在一起的票数+0
        rate = vote_y / 5  # 有效票比例 = （投票的算法数）/总算法数
        if vote_y == 0:  # 有效票数存在为0的可能性
            nag_matrix[m,n] = 0
        else:
            nag_matrix[m, n] = rate * (sum_oppose / vote_y)  # <fi,fj> = 有效票比例 *（支持在一起的算法数/发表意见的算法数） = 支持在一起的算法数/总算法数

# print(nag_matrix)

# 将负-协关联矩阵变成对称矩阵
for m in range(59):
    for n in range(m+1, 60):
        nag_matrix[n, m] = nag_matrix[m,n]
print(nag_matrix)


# 构造协关联矩阵
co_matrix = pos_matrix - nag_matrix
print(co_matrix)

# 计算每个特征的协关联值
co_val = []
for i in range(60):
    val = sum(abs(co_matrix[i,:]))  # 协关联值等于每一个特征与其他特征的关联值（取绝对值）的和
    co_val.append(val)
co_matrix1 = np.insert(co_matrix, co_matrix.shape[0], values=co_val, axis=1)  # 在协关联矩阵后添加协关联值这一列
print(co_matrix1)
co_matrix2 = np.insert(co_matrix1, co_matrix1.shape[1], values=votes, axis=1)  # 在协关联矩阵1后添加协投票数这一列
print(co_matrix2)



# 去掉已选的特征，重新计算特征与其余特征的协关联值
def re_coval(arr,co_matrix,fi):
    new_co_value = []
    for i in range(co_matrix.shape[1]):
        for j in range(len(arr)):
            a = arr[j]
            co_matrix[a,:] = 0
            co_matrix[:,a] = 0
        new = sum(co_matrix[i,:])
        new_co_value.append(new)
    return new_co_value

def re_coval1(arr,co_matrix,fi):
    for j in range(len(arr)):
        a = arr[j]
        co_matrix[a, :] = -500
        co_matrix[:, a] = -500
    new_co_value = sum(abs(co_matrix[fi, :]))
    return new_co_value

print(co_matrix[1])

def feat_extracture(votes,co_matrix,co_matrix2,k):
    feat_list = []
    # 先根据投票数选择第一个特征
    # 获得投票数最多得特征
    max = np.max(votes)  # 最多的票数
    max_index = []  # 获得票数最多的特征
    for i in range(len(votes)):
        if votes[i] == max:
            max_index.append(i)
    if len(max_index) > 1:
        max_co_val = 0
        for i in range(len(max_index)):  # 计算max_index每一个位置的协关联值，选择协关联值最大的位置，即选择协关联值最大的特征
            if(co_matrix2[max_index[i],-2] > max_co_val):
                # 如果恰好有两个特征（fp,fq,p<q）的投票数都是最多且协关联值也相同，默认选择第一个特征（fp）
                max_co_val = co_matrix2[max_index[i],-2]
                fi = i  # 选择fi这个特征
    else:
        fi = max_index[0]
    feat_list.append(fi)

    # 选择其余特征，直到选出k个特征
    while(len(feat_list) <= k):
        # 下面以fi为基准，选择与它协关联值最大的特征作为下一个基准，注意还有可能存在相等的情况
        # 除去刚才已选的特征，返回最大的协关联值
        fi = feat_list[-1]
        print(fi)
        # 与刚选出的特征fi最大的协关联值
        row = co_matrix[fi].tolist()
        # row.pop(int(fi))
        for i in range(len(feat_list)):
            a = feat_list[i]
            row[int(a)] = -500
        print(row)
        max = np.max(row)
        print(max)

        max_index = []  # 与刚选出的特征fi最大的协关联值都是哪些特征
        for j in range(co_matrix.shape[1]):
            if co_matrix[fi, j] == max:
                max_index.append(j)
        print(max_index)
        if len(max_index) > 1:
            max_co_val = 0
            for i in range(len(max_index)):  # 计算max_index每一个位置的协关联值，选择协关联值最大的位置，即选择协关联值最大的特征
                if (re_coval1(feat_list, co_matrix, max_index[i]) > max_co_val):
                    max_co_val = re_coval1(feat_list, co_matrix, max_index[i])
                    fi = max_index[i]  # 选择fi这个特征
                    print(fi)
        else:
            fi = max_index[0]
        feat_list.append(fi)
    return feat_list

# result = feat_extracture(votes,co_matrix,co_matrix2,6)
# print(result)



"""
    根据特征重要性进行特征选择后的效果评估函数
    输入：X：样本的重要特征
         Y：样本类别
    输出：分类效果
    runtime:运行时间
    cm:混淆矩阵
    cr:分类效果
"""
def evaluate_Result(X,Y,fearure_index):
    # 只选择与 y 相关度最高的特征，训练/测试一个支持向量机模型来评估该方法的结果
    fearure_index = np.array(fearure_index)
    X = np.array(X)
    X_Reduced = X[:, fearure_index]
    X_Reduced = StandardScaler().fit_transform(X_Reduced)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X_Reduced, Y, test_size=0.3, random_state=101)
    start = time.process_time()

    # 进行随机森林分类
    trainedforest = RandomForestClassifier(n_estimators=700).fit(X_Train, Y_Train)
    # RandomForestClassifier运行时间
    runtime = time.process_time() - start
    # 开始测试
    predictionforest = trainedforest.predict(X_Test)

    # 测试集混淆矩阵
    cm = confusion_matrix(Y_Test, predictionforest)
    # 测试集结果
    cr = classification_report(Y_Test, predictionforest)

    return runtime, cm, cr

feature_index = [0, 3, 5, 1, 37, 58]
testResult = evaluate_Result(X,Y,feature_index)
print("运行时间：",testResult[0])
print("混淆矩阵：",testResult[1])
print("分类结果：",testResult[2])

