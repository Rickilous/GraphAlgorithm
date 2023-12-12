import sys

import numpy as np
import copy
from scipy.optimize import linear_sum_assignment


def prim_mst(adj_matrix):
    num_vertices = len(adj_matrix)
    # 初始化关键值为无穷大
    key = [float('inf')] * num_vertices
    # 初始化父节点数组为-1
    parent = [-1] * num_vertices
    # 将第一个顶点作为起始点
    key[0] = 0
    # 用于记录已经加入最小生成树的顶点
    mst_set = [False] * num_vertices

    for _ in range(num_vertices):
        # 找到具有最小关键值的顶点
        min_key = float('inf')
        min_index = -1
        for v in range(num_vertices):
            if key[v] < min_key and not mst_set[v]:
                min_key = key[v]
                min_index = v

        # 将选定的顶点加入最小生成树
        mst_set[min_index] = True

        # 更新相邻顶点的关键值和父节点索引
        for v in range(num_vertices):
            if 0 < adj_matrix[min_index][v] < key[v] and not mst_set[v]:
                key[v] = adj_matrix[min_index][v]
                parent[v] = min_index

    return parent


def mst_to_adj_matrix(adj_matrix):
    num_vertices = len(adj_matrix)
    parent = prim_mst(adj_matrix)

    # 构建最小生成树的邻接矩阵
    mst_adj_matrix = [[0] * num_vertices for _ in range(num_vertices)]
    for i in range(num_vertices):
        for j in range(num_vertices):
            if parent[j] == i:
                mst_adj_matrix[i][j] = adj_matrix[i][j]
                mst_adj_matrix[j][i] = adj_matrix[i][j]

    return mst_adj_matrix


# 破圈操作
def des_cir(adjacency_matrix, st):
    n = len(adjacency_matrix)
    # 创建邻接矩阵的副本
    new_matrix = np.copy(adjacency_matrix)

    def circle(matrix, start):
        result = []

        # 递归遍历图，寻找环
        def trace(path, matrix, now):
            if len(result) == 1:
                return
            if now == start and len(path) > 2:
                result.append(list(path))
                return
            for i in range(len(matrix)):
                if matrix[now][i] == 0 or matrix[now][i] == -1 or i in path:
                    continue
                path.append(i)
                trace(path, matrix, i)
                path.pop()

        trace([], matrix, start)
        if len(result) != 0:
            result[0].insert(0, start)
            return result[0]
        return result

    while True:
        # 寻找环
        res = circle(new_matrix, st)
        if len(res) == 0:
            break
        mid_len = 0
        mid_st = -1
        # 寻找环中权重最大的边
        for j in range(len(res) - 1):
            if new_matrix[res[j]][res[j + 1]] > mid_len or (
                    new_matrix[res[j]][res[j + 1]] == mid_len and res[j] + res[j + 1] > res[mid_st] + res[mid_st + 1]):
                mid_len = new_matrix[res[j]][res[j + 1]]
                mid_st = j
        # 将找到的边权设置为-1，表示破圈
        new_matrix[res[mid_st]][res[mid_st + 1]] = new_matrix[res[mid_st + 1]][res[mid_st]] = 0

    return new_matrix  # 返回修改后的权重矩阵


def find(x, graph, match, used, M):
    """
    寻找增广路径的递归函数
    Parameters:
    - x (int): 当前尝试配对的左节点索引
    - graph (list[list]): [N[M]], 是N*M矩阵, 记录左右节点之间是否存在连线
    - match (list[int]): [M], 记录右节点被分配给坐标哪个节点
    - used (list[int]): [M], 记录在本轮配对中某个右节点是否已经被访问过,
        因为每一轮每个右节点只能被访问一次, 否则会被重复配对
    - M (int): 右边节点的个数

    Returns:
    - bool: 是否成功找到增广路径
    """
    for j in range(M):
        # x和j是存在连线 and (j在本轮还没被访问过)
        if graph[x][j] != 0 and not used[j]:
            used[j] = True
            # j还没被分配 or 成功把j腾出来(通过递归, 给j之前分配的左节点成功分配了另外1个右节点)
            if match[j] == -1 or find(match[j], graph, match, used, M):
                match[j] = x
                return True
    return False


def hungarian(biMat):
    """
    Parameters:
    - biMat (numpy.ndarray): 二分图的邻接矩阵

    Returns:
    - Match (list[tuple]): 匹配结果，每个元素是一个元组，表示一对匹配的节点
    - result_matrix (numpy.ndarray): 处理后的邻接矩阵，1表示匹配的边，0表示非匹配的边
    """
    N = biMat.shape[0]
    M = biMat.shape[1]

    graph = np.array(biMat)

    match = [-1 for _ in range(M)]
    count = 0

    for i in range(N):
        used = [False for _ in range(M)]
        if find(i, graph, match, used, M):
            count += 1

    Match = []
    for i in range(len(match)):
        if match[i] != -1:
            Match.append((match[i], i))

    # 构建匹配结果矩阵
    result_matrix = np.zeros_like(graph)
    for pair in Match:
        result_matrix[pair[0]][pair[1]] = 1

    return Match, result_matrix


def kuhn_munkres(adj_matrix):
    # 使用linear_sum_assignment函数找到最大权匹配
    row_ind, col_ind = linear_sum_assignment(adj_matrix, maximize=True)
    # 创建一个新的邻接矩阵表示匹配
    matching_matrix = np.zeros_like(adj_matrix)
    matching_matrix[row_ind, col_ind] = 1
    return matching_matrix


def Floyd(Matrix):
    n = Matrix.shape[0]
    dist = Matrix.copy()  # 初始化为无穷大
    e = (dist != -1)
    dist = np.where(e, dist, 999999)  # -1改9999

    path = -1 * np.ones((n, n))  # -1 表示没有路径

    # 兼容性修改
    #
    # e = (dist != -1)
    # dist = np.where(e, dist, 999999)
    # e = (path != 999999)
    # path = np.where(e, path, -1)

    # Floyd 算法主体部分，枚举所有结点对
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # 如果从 i 到 j 经过 k 可以缩短路径，则更新 dist 和 path
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    path[i][j] = k

    return dist, path


def floydwarshall(adMat):
    # 输入的adMat为-1 和0 短路的矩阵
    # 这个算法中对角线和下三角矩阵都为65535，兼容处理为0和-1
    e = (adMat > 0)
    matrix = np.where(e, adMat, 65535)
    matrix_info = np.array(matrix)
    n = matrix_info.shape[0]
    for k in range(n):
        i = 1
        j = 1
        for i in range(n):
            for j in range(n):
                if (matrix_info[i][j] > matrix_info[i][k] + matrix_info[k][j]):
                    matrix_info[i][j] = matrix_info[i][k] + matrix_info[k][j]

    e = (matrix_info != 65535)
    matrix_info = np.where(e, matrix_info, -1)

    # 有向图距离矩阵转为邻接矩阵
    n = matrix_info.shape[0]
    for i in range(n - 1):
        for j in range(n - 1):
            if matrix_info[i + 1][n - j - 1] != -1 and matrix_info[i][n - j - 2] != -1:
                matrix_info[i][n - j - 1] = -1
    for i in range(n):
        matrix_info[i][i] = 0
    return matrix_info


def dijkstra(adMat):
    # 慢就慢吧，心情好了再来优化哈哈哈哈

    # 假设输入的矩阵满足条件：排在第一的节点可以抵达其他任意节点

    # 判断是否为无向图，用对称矩阵判断
    flag = 0  # 无向图
    for i in range(len(adMat)):
        for j in range(i + 1, len(adMat)):
            if adMat[i][j] != adMat[j][i]:
                flag = 1  # 有向图/非无向图
                break

    max = np.max(adMat)
    print('max:', max)
    S = []
    N = list(range(len(adMat)))
    e = (adMat > 0)
    adMat = np.where(e, adMat, 0)
    S.append(N.pop(0))
    finally_adMat = np.zeros(adMat.shape)
    while (len(N) != 0):
        for i in S:
            for j in S:
                adMat[i][j] = 0
        x = 0
        y = 0
        w = max + 1
        for i in S:
            for j in N:
                if adMat[i][j] > 0 and adMat[i][j] < w:
                    x = i
                    y = j
                    w = adMat[i][j]
        finally_adMat[x][y] = w
        S.append(N.pop(N.index(y)))

    if flag == 0:
        for i in range(len(finally_adMat)):
            for j in range(i, len(finally_adMat)):
                if finally_adMat[i][j] == 0:
                    finally_adMat[i][j] = finally_adMat[j][i]
                else:
                    finally_adMat[j][i] = finally_adMat[i][j]
    print(finally_adMat)
    return finally_adMat

'''
最小生成树算法kruskal
'''
def takeThrid(elem):
    '''
    把第三列作为排序键，配合adMat2adList函数使用
    :param elem:自动获取
    :return:无
    '''
    return elem[2]


def adMat2adList(adMat):
    '''
    邻接矩阵转为邻接表
    :param adMat:邻接矩阵
    :return:按权值从小到大排序的邻接表
    '''
    adList = []
    nodeNum = len(adMat)
    for i in range(nodeNum):
        for j in range(i + 1, nodeNum):
            if adMat[i][j] > 0:
                adList.append([i, j, adMat[i][j]])
    adList.sort(key=takeThrid)
    return adList




def findRoot(node, parentList):
    t= node
    while parentList[t]>-1:
        t=parentList[t]
    return t

def kruskal(adMat):
    '''
    克鲁斯卡尔算法主函数
    :param adMat: 输入的邻接矩阵 -1元素表示i和j无边
    :return: 最小生成树的邻接矩阵
    '''
    print(adMat)
    n=adMat.shape[0]
    # 排序邻接表
    adList = adMat2adList(adMat)
    # 每个节点所在树的的根节点
    parentList = [-1 for i in range(n)]

    # 结果邻接矩阵初始化为全0
    endMat = np.zeros(adMat.shape)

    # 尝试把权值小的边加入进来
    num=0
    for t in adList:
        print("判断边:", t)
        root_i = findRoot(t[0], parentList)
        root_j = findRoot(t[1], parentList)
        print(root_i , root_j)
        # if haveCircle(endMat):
        if root_i != root_j:
            print('接纳边:', t)
            num+=1
            parentList[root_j] = root_i
            endMat[t[0]][t[1]] = t[2]  # 0和1号元素是边t的起点和终点，2号元素是边t上面的权重
            endMat[t[1]][t[0]] = t[2]
            if num == n-1:
                return endMat
def runKruskal(input_Mat):
    print("runKruskal is done,now output")
    output_Mat = kruskal(input_Mat)
    return output_Mat


def runPrim(input_Mat):
    output_Mat = mst_to_adj_matrix(input_Mat)
    return output_Mat


def runDes_Cir(input_Mat):
    output_Mat = des_cir(input_Mat, 0)
    return output_Mat


def runDijkstra(input_Mat):
    output_Mat= dijkstra(input_Mat)
    print(output_Mat)
    return output_Mat


def runFloyd(input_Mat):
    output_Mat, path = Floyd(input_Mat)
    return output_Mat, path


def runFloydwarshall(input_Mat):
    output_Mat= floydwarshall(input_Mat)
    return output_Mat


def runHungarian(input_Mat):
    Match, output_Mat = hungarian(input_Mat)
    print(output_Mat)
    return output_Mat


def runKuhn_Munkres(input_Mat):
    output_Mat = kuhn_munkres(input_Mat)
    return output_Mat
