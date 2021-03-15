import numpy as np
from PIL import Image
from pylab import *

from clustering_ensembled_1 import tri
print("列数：", tri.shape[1])
"""
    测量样本稳定性
    s(xi)=1/n*sum(pi1+pi2+...+pin)
"""
s = []
def stablity(matr):
    n = matr.shape[0]
    for i in range(matr.shape[0]):
        s1 = 1/n * sum(matr[i,:])
        s.append(s1)
    return s
print(stablity(tri))
s = list(s)

"""
    基于线性函数的稳定性
    fl(pij)=abs((pij - t)/t)  if pij<t
    fl(pij)=abs((pij-t)/(1-t)) if pij>t
    sl(xi) = 1/n*sum(fl(pi1)+fl(pi2)+...+fl(pin))
"""
sl = []
def stablity_line(matr,t):
    n = matr.shape[0]
    for i in range(matr.shape[0]):
        sum_fl = 0
        for j in range(matr.shape[1]):
            if matr[i,j] < t:
                fl = abs((matr[i,j]-t)/t)
            else:
                fl = abs((matr[i,j]-t)/(1-t))
            sum_fl += fl
        s = (1/n) * sum_fl
        sl.append(s)
    return sl
print(stablity_line(tri,0.5))
sl = list(sl)

"""
    基于二次函数的稳定性
    fq(pij)=np.power((pij - t)/t),2)   if pij<t
    fq(pij)=np.power((pij - t)/(1-t)),2) if pij>t
    sq(xi) = 1/n*sum(fq(pi1)+fq(pi2)+...+fq(pin))
"""
sq = []
def stablity_quadratic(matr,t):
    n = matr.shape[0]
    for i in range(matr.shape[0]):
        sum_fq = 0
        for j in range(matr.shape[1]):
            if matr[i, j] < t:
                fq = np.power(((matr[i, j] - t)/t),2)
            else:
                fq = np.power(((matr[i, j] - t)/(1-t)),2)
            sum_fq += fq
        s = (1 / n) * sum_fq
        sq.append(s)
    return sq
print(stablity_quadratic(tri,0.4))
sq = list(sq)

"""
    输入稳定值利用最大类间方差法求阈值t
"""
sigama = []
def find_t(s):
    sum0 = 0
    n = len(s)
    miu = sum(s) / n
    for i in range(len(s)):
        sum0 += s[i]
        sum1 = sum(s) - sum0
        w0 = (i+1)/n
        w1 = 1-((i+1)/n)
        miu0 = sum0/(i+1)
        miu1 = sum1/(len(s)-(i+1))
        sigama_t = w0 * np.power((miu0 - miu),2) + w1 * np.power((miu1 - miu),2)
        sigama.append(sigama_t)
    return (sigama)

print(find_t(sl))
t = max(find_t(sl))
print("阈值：", t)

# ss1 = s[0:3]
# print(ss1)
# n = 3
# sum0 = sum(ss1)
# w0 = 3/99
# w1 = 1 - (3/99)
# miu0 = sum0/3
# sum1 = sum(s)-sum0
# miu1  = sum1/96
# miu = sum(s)/99
# sigama_t = w0 * np.power((miu0 - miu),2) + w1 * np.power((miu1 - miu),2)
# print(sigama_t)