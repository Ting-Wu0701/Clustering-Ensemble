"""
A design to calculating the distance of enumerating discrete variables
    DiscreteDistance, more like a distance, for ref,... todo: reference
In function test(), I gave a simple but not completed way of estimation.
    todo 三角不等式的完备性检查
by Yihang Hao
Jan.27th,2020
v1.2

"""
import math
from itertools import permutations, combinations

import numpy as np


def getDiscreteDistance(_arr1, _arr2, _size, coefficientMatrix):
    """
    Passing in 2 sets of permutations, the first permutation is BASE by default,
        and the corresponding elements are encoded as 1,2,3,4,5,...
    Then, the reverse order number determined by the number is calculated
        to obtain the discreteDistance
    :param _arr1:   Discrete_set_1
    :param _arr2:   Discrete_set_2
    :param _size:   Make sure the size of the set is corresponded
    :return: the calculated distance between two sets
        todo: To deal with the uniqueness of the reverse number,
    """
    DiscreteDistance = 0
    kvDictList = []  # Dict List, determined by _arr1 where nums are 1,2,3,4..._size，
    for i in range(0, _size):
        kvDictList.append({'item': _arr1[i], 'num': i})
    tar = []  # Interpret _arr2 to "0,3,1,4,2" according to kvDict, record into tar[]
    for i in range(0, _size):
        # Find the corresponding position of i_th element in _arr2 in kvDictList
        for j in range(0, _size):
            # if found, copy the letter into tar[]
            if _arr2[i] == kvDictList[j]['item']:
                tar.append(kvDictList[j]['num'])
    DiscreteDistance = getInverse(tar)
    return DiscreteDistance


def getInverse(arr, coefficientMatrix=[5, 1, 1, 1, 1]):
    """
        利用逆序数计算排序成本cost, cost=∑逆数×逆子
    :param coefficientMatrix: 试错的系数矩阵
    :param arr: 传进来的一个离散集合
    :return: 返回计算的cost值
    """
    cost = 0
    for i in range(len(arr)):
        step = 0
        for j in range(i):
            if arr[j] > arr[i]:
                """
                    cost = a_4x^4+a_3x^3+a_2x^2+a_1x+a0, where x = a[j]
                """
                for k in range(0, 5):
                    cost += 0.1 * coefficientMatrix[k] * pow(arr[j], k)
                # cost += arr[j]
                # cost += math.cos(arr[j] / 4) + 1
                # Multiple by accumulation
                # cost += math.exp(arr[j])
    return cost


def lookForMatrix():
    MAX_TRY = 100
    AvailableList = []
    flag = True
    Current_Matrix = [1, 1, 1, 1, 1]
    for i in range(1, 5):
        for j in range(0, MAX_TRY):
            for k in range(0, MAX_TRY):
                for w in range(0, MAX_TRY):
                    for v in range(0, MAX_TRY):
                        print("Current_Matrix:{}".format(Current_Matrix))
                        # 先做计算出距离矩阵

                        MyDistanceMatrix = doCalculation(Current_Matrix)
                        f1 = duplicateCheck(MyDistanceMatrix)
                        f2 = triangleCheckByMatrix(MyDistanceMatrix)
                        flag = f1 and f2
                        if flag:
                            AvailableList.append(Current_Matrix)
                            file = open("data.txt", 'a')
                            print(Current_Matrix, file=file)
                            file.close()
                        Current_Matrix = [v,w,k,j,i]
    for line in AvailableList:
        print(AvailableList)
    return AvailableList


def doCalculation(coefficientMatrix):
    """
    using functions above to make out a distance-like distance matrix.
    example:
        [[0,1,2,3],
         [1,0,2,3],
         [2,2,0,7],
         [3,3,7,0]]
    :return: distance Matrix calculated
    """
    list_C = ['A', 'B', 'C', 'D']  # todo: here we can do any number of enum items
    ArrangeList = list(permutations(list_C))
    DistanceMatrix = [[0 for i in range(len(ArrangeList))] for i in range(len(ArrangeList))]
    for i in range(0, len(ArrangeList)):
        for j in range(0, len(ArrangeList)):
            temp = getDiscreteDistance(ArrangeList[i], ArrangeList[j], len(list_C), coefficientMatrix)
            DistanceMatrix[i][j] = temp
    # output to a result_x.csv
    # writeToCsv(DistanceMatrix)
    return DistanceMatrix


def duplicateCheck(_distMatrix):
    """
    check whether a distance matrix has duplicate value appears in the same item
    Example: from item (a,b,d,c) to any other items like (a,b,c,d) or
            (c,a,b,d) where distances should be varied from each other.
    :param _distMatrix: Matrix to be check
    :return: flag reflect to whether the check is passed
    """
    dupFlag = True
    duplicateResult = []
    for line in _distMatrix:
        for i in line:
            if line.count(i) > 1:
                duplicateResult.append(line)
                dupFlag = False
                break
    if dupFlag:
        print("重复检查:通过")
    else:
        print("重复检查:不通过\n重复的行如下：")
        for it in duplicateResult:
            print(it)
    return dupFlag


def triangleCheckByList(_len, _detailFlag=False):
    """
    Verification method of triangle-inequality rules as a basic rule of distance definition
            by list, only to check if this 3 lengths are validated.
    :param _detailFlag: whether print results of every combine items
    :param _len: a array, records the 3 side lengths of a triangle
    :return: whether _len array satisfy triangle-inequality
    """
    combineList = list(combinations(_len, 3))
    # analyze respectively to different combinations of the three side lengths
    for item in combineList:
        flag = True
        list(item).sort()  # Sort three values
        if item[0] + item[1] <= item[2] and item[2] - item[1] >= item[0]:
            flag = False

        if _detailFlag:
            print(item, end='\t')
            print(flag)
    return flag


def triangleCheckByMatrix(_distMatrix, _detailFlag=False):
    """
    Verification method of triangle-inequality rules as a basic rule of distance definition
        by Matrix, a comprehensive test.
    :param _detailFlag: whether print results of every combine items
    :param _distMatrix: matrix to be tested
    :return: todo return a flag
    """
    flag = True
    size = len(_distMatrix[0])
    step = [0, 0, 0]
    for i in range(0, size):
        for j in range(0, size):
            if j == i:
                continue
            step[0] = _distMatrix[i][j]
            for k in range(0, size):
                if j == k or k == i:
                    continue
                step[1] = _distMatrix[j][k]
                step[2] = _distMatrix[i][k]
                if not triangleCheckByList(step, _detailFlag):
                    flag = False
                    # print("不满足三定律的边长为：", end='')
                    # print("[{},{},{}]".format(step[0], step[1], step[2]))
                    break
    if flag:
        print("三角检查：通过")
    else:
        print("三角检查：不通过")
    return flag


def writeToCsv(_list, _filename='result_x.csv'):
    import csv
    line = 0
    with open(_filename, "w", newline='') as file:
        writer = csv.writer(file)
        for data in _list:
            line += 1
            writer.writerow(data)
            print("{}have been written".format(line))
            break


if __name__ == '__main__':
    WRITE_TO_CSV_FLAG = False
    DETAIL_FLAG = True

    ifFoundFlag = False  # 找到解了咩？

    lookForMatrix()


