# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!
import os
from time import sleep

import matplotlib.pyplot as plt
from matplotlib.pyplot import cla,savefig
import networkx as nx
import numpy as np
from PyQt5.QtCore import QRect,QMetaObject,QCoreApplication
from PyQt5.QtWidgets import QWidget,QGroupBox,QLabel,QPushButton,QTextEdit,QTextBrowser,QStatusBar
from PyQt5.QtGui import QPixmap
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets
import copy


class Ui_MainWindow(object):

    '''全局数据'''

    global_strGraph = None
    input_Mat = None
    output_Mat = None
    M=None

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(924, 588)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(40, 20, 321, 251))
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 20, 301, 221))
        self.label.setObjectName("label")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(40, 290, 321, 251))
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 301, 221))
        self.label_2.setObjectName("label_2")

        '''连接获取图按钮'''
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(640, 30, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.showInput)


        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(640, 80, 241, 121))
        self.groupBox_3.setObjectName("groupBox_3")

        '''连接破圈法'''
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_2.setGeometry(QtCore.QRect(10, 70, 111, 26))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.runDes_Cir)

        '''连接Kruskal算法'''
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_3.setGeometry(QtCore.QRect(10, 30, 111, 26))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_3.clicked.connect(self.runKruskal)

        '''连接Prim算法'''
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_4.setGeometry(QtCore.QRect(130, 30, 111, 26))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_4.clicked.connect(self.runPrim)

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(640, 220, 241, 71))
        self.groupBox_4.setObjectName("groupBox_4")


        '''链接hungarin算法'''
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_5.setGeometry(QtCore.QRect(10, 30, 111, 26))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_5.clicked.connect(self.runHungarin)


        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(640, 400, 241, 131))
        self.groupBox_5.setObjectName("groupBox_5")\


        '''连接FLoyd算法'''
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_8.setGeometry(QtCore.QRect(10, 30, 111, 26))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.clicked.connect(self.runFloyd)

        '''链接Dijkstra算法'''
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_9.setGeometry(QtCore.QRect(130, 30, 111, 26))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.clicked.connect(self.runDijkstra)


        '''链接Floyd Warshell算法'''
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_5)
        self.pushButton_6.setGeometry(QtCore.QRect(10, 70, 111, 26))
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_6.clicked.connect(self.runFloydwarshall)


        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(370, 20, 261, 251))
        self.groupBox_6.setObjectName("groupBox_6")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_6)
        self.textEdit.setGeometry(QtCore.QRect(10, 20, 241, 221))
        self.textEdit.setObjectName("textEdit")
        self.groupBox_7 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_7.setGeometry(QtCore.QRect(370, 290, 261, 251))
        self.groupBox_7.setObjectName("groupBox_7")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox_7)
        self.textBrowser.setGeometry(QtCore.QRect(10, 20, 241, 221))
        self.textBrowser.setObjectName("textBrowser")
        self.groupBox_8 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_8.setGeometry(QtCore.QRect(640, 310, 241, 71))
        self.groupBox_8.setObjectName("groupBox_8")


        '''链接Kuhn Munkres算法'''
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_8)
        self.pushButton_7.setGeometry(QtCore.QRect(10, 30, 111, 26))
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_7.clicked.connect(self.runkuhn_Munkres)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    ######################################
    ############    算法    ###############
    ######################################
    '''最小生成树——kruskal'''
    def runKruskal(self):
        self.output_Mat=self.kruskal(self.input_Mat)
        self.pushImg('output')
    def takeThrid(self, elem):
        '''
        把第三列作为排序键，配合adMat2adList函数使用
        :param elem:自动获取
        :return:无
        '''
        return elem[2]
    def adMat2adList(self, adMat):
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
        adList.sort(key=self.takeThrid)
        return adList
    def adMat2raList(self, adMat):
        '''
        邻接矩阵转为用list表示的可达链表
        :param adMat:邻接矩阵
        :return:用list表示的可达链表，若用在无向图中会包含两个方向
        '''
        raList = []  # raList[i]包含了节点i能到达的所有节点
        for i in range(adMat.shape[0]):
            raList.append([])
            for j in range(adMat.shape[1]):
                if adMat[i][j] > 0: raList[i].append(j)  # i→j有路
        return raList
    def haveCircle(self, adMat):
        '''
        根据邻接矩阵判断是否有圈，邻接矩阵代表的图可以是非连同图
        :param adMat:邻接矩阵
        :return:0没有圈，1有圈
        '''
        # print("中间结果：",adMat)
        raList = self.adMat2raList(adMat)  # 矩阵转可达链表
        print("raList", raList)
        visited = np.zeros(adMat.shape[0])  # 初试化所有点没有访问过
        isCircle = 0  # 假设没有圈
        for index in range(visited.shape[0]):  # 这层循环用于搜索还未访问过的图
            stack = []  # 访问栈，记录接下来的访问顺序，访问结束即出栈
            if visited[index] == 1:  # 如果当前访问的节点已经访问过，表示这个节点属于之前已经访问过的分图，跳过
                continue
            isCircle, visited = self.MultiBFS(index, visited, raList, stack)  # 运行多重广度有线遍历算法，查找有没有圈
            if isCircle:
                break
        return isCircle
    def MultiBFS(self, index, visited, raList, stack):
        '''
        多重广度优先遍历算法，配合haveCircle函数寻找圈用
        因为克鲁斯卡尔算法中间步骤的图可能连同分量是大于1的，本算法可以应对此情况。
        原理为每遍历完一个图，再去遍历下一个图
        :param index:当前访问节点的标号
        :param visited:已访问节点的标号
        :param raList:用list表示的可达链表
        :param stack:表示访问顺序的栈
        :return:1.是否有圈；2.运行后的访问记录
        '''
        if len(stack) == 0: stack.append(index)  # 如果栈是空的，说明算法为第一次递归、或者上一次递归遍历完了一个分图，需要加入一个未被访问的节点到栈里
        print("......访问到%d节点......" % stack[0])
        # print("stack:",stack)
        if visited[stack[0]] == 1:  # 即将访问的节点已经被访问过，有圈，终止算法
            print("圈")
            return 1, visited
        visited[stack[0]] = 1  # 标记当前节点为已被访问
        # print("visited:",visited)

        for next in raList[stack[0]]:
            # print("即将添加：",raList[stack[0]])
            stack.append(next)
            # print("stack:",stack)
            raList[next].pop(raList[next].index(stack[0]))

        index = stack.pop(0)
        # print("stack:", stack)
        if len(stack) == 0:
            print("........终点是%d节点......" % index)
            # print('\n\n')
            return 0, visited

        return self.MultiBFS(index, visited, raList, stack)
    def kruskal(self, adMat):
        print(adMat)

        # 排序邻接表
        adList = self.adMat2adList(adMat)

        # 结果矩阵
        endMat = np.zeros(adMat.shape)

        # 记录某点经过次数
        for t in adList:
            print("正在判断边：", t)
            endMat[t[0]][t[1]] = t[2]
            endMat[t[1]][t[0]] = t[2]
            if self.haveCircle(endMat):
                print('拒绝边', t)
                endMat[t[0]][t[1]] = 0
                endMat[t[1]][t[0]] = 0
                print('\n')
                continue
            print('接纳边', t)
            print('\n')

        return endMat


    '''最小生成树——Prim'''
    def runPrim(self):
        self.output_Mat=self.Prim(self.input_Mat)
        self.pushImg('output')
    def Prim(self,Matrix):
        MAX=999999
        nodeNum = Matrix.shape[0]
        start = 0
        marked_vertexes = [start]  # 记录已经存在与最小生成树的节点
        # 将每一个节点分开存放方便遍历
        origin_vertext = [i for i in range(1, nodeNum)]
        weight = 0  # 总权重
        edges_weight = []  # 结果列表
        Tree_Matrix = []
        for i in range(nodeNum):
            Tree_Matrix.append([])
            for j in range(nodeNum):
                Tree_Matrix[i].append(MAX)
        for i in range(nodeNum):
            for j in range(nodeNum):
                Tree_Matrix[i][i] = 0
        print(Tree_Matrix)

        while len(origin_vertext) > 0:
            vertex1, vertex2 = 0, 0
            temp = MAX
            for i in marked_vertexes:  # 遍历已经存在于最小生成树的节点
                for j in origin_vertext:  # 遍历每不存在于最小生成树的节点
                    if Matrix[i][j] == -1 or Matrix[i][j] == 0:  # 如果连两个节点不连通，继续循环
                        continue
                    else:  # 如果联通
                        if temp > Matrix[i][j]:  # 选择存在于最小生成树和不存在于最小生成树的两个节点中边权最小的节点
                            temp = Matrix[i][j]  # 中间变量存储最小边权
                            vertex1 = j  # 中间变量不存在于最小生成树的点
                            vertex2 = i  # 中间变量存在于最小生成树的点
            Tree_Matrix[vertex2][vertex1] = temp
            weight += temp  # 计算总边权
            edges_weight.append([vertex2, vertex1, temp])  # 将此节点存于结果中
            marked_vertexes.append(vertex1)  # 将不存在于最小生成树的节点插入到最小生成树的节点
            origin_vertext.remove(vertex1)  # 在不存在于最小生成树的列表中删除此节点
        edges_weight.append(weight)  # 将总边权插入到最终结果
        print('edges_weight', edges_weight)
        print('Tree_Matrix', Tree_Matrix)

        # 结果兼容处理
        returnMat = np.zeros(Matrix.shape)
        e = (returnMat != 0)
        returnMat = np.where(e, returnMat, -1)
        for e in edges_weight[0:-1]:
            i, j, k = e
            if i > j:
                temp = i
                i = j
                j = temp
            print(i, '→', j, '=', k)
            returnMat[i][j] = k
        for n in range(nodeNum):
            returnMat[n][n] = 0
        return returnMat


    '''最小生成树——破圈法'''
    def runDes_Cir(self):
        self.output_Mat=self.des_cir(self.input_Mat,0)
        self.pushImg('output')
    def circle(self,tupo, st):
        result = []

        def trace(path, tupo, now):
            if len(result) == 1: return
            if now == st and len(path) > 2:
                result.append(list(path))
                return
            for i in range(len(tupo)):
                if tupo[now][i] == 0 or tupo[now][i] == -1 or i in path: continue
                path.append(i)
                trace(path, tupo, i)
                path.pop()

        trace([], tupo, st)
        if len(result) != 0:
            result[0].insert(0, st)
            return result[0]
        return result
    def des_cir(self,tupo, st):
        n = len(tupo)
        for i in range(n):
            while (1):
                res = self.circle(tupo, st)
                if len(res) == 0: break
                mid_len = 0
                mid_st = -1
                for j in range(len(res) - 1):
                    if tupo[res[j]][res[j + 1]] > mid_len or (
                            tupo[res[j]][res[j + 1]] == mid_len and res[j] + res[j + 1] > res[mid_st] + res[
                        mid_st + 1]):
                        mid_len = tupo[res[j]][res[j + 1]]
                        mid_st = j
                tupo[res[mid_st]][res[mid_st + 1]] = tupo[res[mid_st + 1]][res[mid_st]] = -1
        adMat = []
        for item in tupo:
            adMat.append(item)
        adMat = np.array(adMat)
        return adMat

    '''最短路径——dijkstra'''
    def runDijkstra(self):
        self.output_Mat=self.dijkstra(self.input_Mat)
        print('结果：',self.output_Mat)
        self.pushImg('output')
    def dijkstra(self, adMat):
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
        print(S, N)
        finally_adMat = np.zeros(adMat.shape)
        print(adMat)
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
            print(S, N)

        if flag == 0:
            for i in range(len(finally_adMat)):
                for j in range(i, len(finally_adMat)):
                    if finally_adMat[i][j] == 0:
                        finally_adMat[i][j] = finally_adMat[j][i]
                    else:
                        finally_adMat[j][i] = finally_adMat[i][j]

        return finally_adMat


    '''最短路径——floyd'''
    def runFloyd(self):
        self.output_Mat=self.Floyd(self.input_Mat)
        self.pushImg('output')
    def Floyd(self,Matrix):
        # 兼容操作
        e = (Matrix != -1)
        Matrix = np.where(e, Matrix, 999999)

        # 主要部分
        vertex_counts = len(Matrix)
        for k in range(0, vertex_counts):
            for i in range(0, vertex_counts):
                for j in range(0, vertex_counts):
                    if Matrix[i][j] > Matrix[i][k] + Matrix[k][j]:
                        Matrix[i][j] = Matrix[i][k] + Matrix[k][j]

        # 兼容操作
        e = (Matrix != 999999)
        Matrix = np.where(e, Matrix, -1)
        n = Matrix.shape[0]
        for i in range(n):
            for j in range(i, n):
                Matrix[j][i] = -1
        print(Matrix)
        for i in range(n - 1):
            for j in range(n - 1):
                if Matrix[i + 1][n - j - 1] != -1 and Matrix[i][n - j - 2] != -1:
                    Matrix[i][n - j - 1] = -1
        for i in range(n):
            Matrix[i][i] = 0
        return Matrix


    '''最短路径——floyd-warshall'''
    def runFloydwarshall(self):
        self.output_Mat = self.floydwarshall(self.input_Mat)
        print('结果：', self.output_Mat)
        self.pushImg('output')
    def floydwarshall(self,adMat):
        #输入的adMat为-1 和0 短路的矩阵
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
            matrix_info[i][i]=0

        self.finally_AdMat=matrix_info

        return matrix_info

    '''最大匹配——hungarin'''
    def runHungarin(self):
        print('匈牙利算法输入:',self.input_Mat)
        M=self.hungarin(self.input_Mat)
        self.M=M
        print('匹配：',M)
        self.output_Mat=np.array(self.M2MAT(self.input_Mat,M),dtype=np.int)
        print("结果矩阵：",self.output_Mat)
        self.pushImg('output')
    def find(self,x, graph, match, used, M):
        """
        x (int): 当前尝试配对的左节点索引
        graph (list[list]): [N[M]], 是N*M矩阵, 记录左右节点之间是否存在连线
        match (list[int]): [M], 记录右节点被分配给坐标哪个节点
        used (list[int]): [M], 记录在本轮配对中某个右节点是否已经被访问过,
            因为每一轮每个右节点只能被访问一次, 否则会被重复配对
        """
        for j in range(M):
            # x和j是存在连线 and (j在本轮还没被访问过)
            if graph[x][j] == 1 and not used[j]:
                used[j] = True
                # j还没被分配 or 成功把j腾出来(通过递归, 给j之前分配的左节点成功分配了另外1个右节点)
                if match[j] == -1 or self.find(match[j], graph, match, used, M):
                    match[j] = x
                    return True
        return False
    def hungarin(self,biMat):
        """
        N和M分别代表左右边节点的个数,
        edges代表节点之间的连线
        graph是N*M矩阵, 记录左右节点之间是否存在连线
        """
        # 兼容处理
        N = biMat.shape[0]
        M = biMat.shape[0]
        '''
        edges = [(1, 0), (2, 0), (0, 1), (1, 1), (3, 1), (4, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 4), (4, 4)]
        graph = []
        for i in range(N):
            graph.append([])
            for j in range(M):
                if (i, j) in edges:
                    graph[i].append(1)
                else:
                    graph[i].append(0)
        '''
        graph = np.array(biMat)
        print("初始图: ")
        for i in range(N):
            print(graph[i])
        print("")

        # match记录左边节点最终与左边哪个节点匹配
        match = [-1 for _ in range(M)]
        # count记录最终的匹配数
        count = 0
        # 遍历左节点, 依次尝试配对左边每个节点,
        # 对于每次尝试配对的左节点,
        # 如果能在右边找到与之匹配的右节点
        # 则匹配数+1
        for i in range(N):
            # 每一轮是一次全新的查找, 所以used要重置,
            # 但是是基于前面几轮找到的最优匹配, 所以match是复用的
            used = [False for _ in range(M)]
            if self.find(i, graph, match, used, M):
                count += 1

        print("最大匹配个数: ", count)
        print("右节点匹配到的左节点序号: ", match)

        # 兼容处理
        Match = []
        print(match)
        for i in range(len(match)):
            if match[i] != -1:
                Match.append((match[i], i))

        return Match  # 即return M

    '''最优匹配——kuhn-munkres'''
    def runkuhn_Munkres(self):
        print('输入:',self.input_Mat)
        M=self.kuhn_munkres(self.input_Mat,type='MAX')
        self.M = M
        print('self.M:',self.M)
        print('成功获取KM算法匹配',M)
        self.output_Mat=np.array(self.M2MAT(self.input_Mat,M),dtype=np.int)
        print('成功转换匹配M为矩阵',self.output_Mat)
        self.pushImg('output')
    def kuhn_munkres(self,biMat,type='MAX'):
        '''
        KM算法，这是不依赖匈牙利算法的版本
        可以实现最大匹配或最小匹配，因为两者可以相互转换
        :param biMat:二部图的邻接矩阵
        :param type: type==MAX，表示需要用最小匹配求最大匹配；若type==MIN，则求最小匹配
        :return:X到Y的匹配
        '''

        origin_biMat = copy.deepcopy(biMat)

        # 如果是最大匹配转换为最小匹配
        if type == 'MAX':
            biMat = -biMat

        print(biMat)

        '''1、行列减去最小值'''
        for i in range(biMat.shape[0]):
            m = np.min(biMat[i])
            biMat[i] = biMat[i] - m
        for i in range(biMat.shape[1]):
            m = np.min(biMat[:, i])
            biMat[:, i] = biMat[:, i] - m

        print('初试Mat:', biMat)

        '''2、循环'''
        # 1:用最少的的线覆盖所有0
        # 求边矩阵

        X = list(range(biMat.shape[0]))
        Y = list(range(biMat.shape[1]))

        while True:
            a = np.where((biMat == 0), biMat, 1) + 1  # (条件，操作矩阵，不满足条件的全部赋值)
            a = np.where((a == 1), a, 0)
            xs0 = np.zeros(shape=a.shape[0])  # 行的0数
            ys0 = np.zeros(shape=a.shape[1])  # 列的0数
            xe = []  # x方向上线的索引
            ye = []  # y方向上线的索引

            print('0的位置:', a)

            # 定位线坐标
            while (a.sum() != 0):
                for i in range(a.shape[0]):
                    xs0[i] = np.sum(a[i, :])
                for i in range(a.shape[1]):
                    ys0[i] = np.sum(a[:, i])
                maxx = max(xs0)
                maxy = max(ys0)

                if maxx >= maxy:
                    index = list(xs0).index(maxx)
                    xe.append(index)
                    a[index] = 0
                if maxx < maxy:
                    index = list(ys0).index(maxy)
                    a[:, index] = 0
                    ye.append(index)

            xe.sort()
            ye.sort()

            print(len(xe) + len(ye), '==', biMat.shape[0])

            if (len(xe) + len(ye)) == biMat.shape[0]:
                break

            X_xe = list(set(X) - set(xe))
            Y_ye = list(set(Y) - set(ye))

            print('X线:', xe)
            print('Y线:', ye)
            print('X补:', X_xe)
            print('Y补:', Y_ye)

            m = np.min(biMat[np.ix_(X_xe, Y_ye)])
            biMat[np.ix_(X_xe, Y_ye)] = biMat[np.ix_(X_xe, Y_ye)] - m
            biMat[np.ix_(xe, ye)] = biMat[np.ix_(xe, ye)] + m

        '''3、根据当前的矩阵，选则匹配，优先匹配度数小的点'''
        print('最后结果：\n', biMat)
        a = np.where((biMat == 0), biMat, 1) + 1  # (条件，操作矩阵，不满足条件的全部赋值)
        a = np.where((a == 1), a, 0)
        M = []

        while (a.sum() != 0):
            for i in range(a.shape[0]):
                xs0[i] = np.sum(a[i, :])
            for i in range(a.shape[1]):
                ys0[i] = np.sum(a[:, i])

            n = 0
            index = -1
            d = -1
            for i in range(len(xs0)):
                if (n == 0 or (xs0[i] > 0 and n > xs0[i])):
                    index = i
                    n = xs0[i]
                    print('x,n=', n)
                    d = 0
                if (n == 0 or (ys0[i] > 0 and n > ys0[i])):
                    index = i
                    n = ys0[i]
                    print('y,n=', n)
                    d = 1

            print("min:", n)
            print("方向:", d)
            print("索引:", index)

            if d == 0:
                print('-' * 30, a[index])
                index2 = list(a[index]).index(1)
                M.append((index, index2))
                a[index] = 0
                a[:, index2] = 0
            if d == 1:
                index2 = list(a[:, index]).index(1)
                M.append((index2, index))
                a[:, index] = 0
                a[index2] = 0

        return M




    ######################################
    ############    操作    ###############
    ######################################
    '''匹配转二部矩阵'''
    def M2MAT(self,Mat,M):
        '''
        把矩阵中被匹配覆盖的边分离出来
        :param mat: 二部矩阵
        :param M: 匹配集
        :return: 匹配矩阵
        '''
        Mat2=np.zeros(shape=Mat.shape)
        print(Mat2)
        for m in M:
            Mat2[m[0]][m[1]]=Mat[m[0]][m[1]]
        print('↓'*20)
        print(Mat2)
        return Mat2

    '''获取图的结构化信息'''
    def getGraph(self):
        '''
        从输入框解析出图的信息，并结构化存储到全局支持有向图、无向图、二部图
        只负责从文本框中解析出图的信息并保存到全局数据
        :return:无
        '''
        s = self.textEdit.toPlainText()
        strGraph = s.split('\n')
        print("获取：",strGraph)

        #二部图和其他图要分开处理：
        Mat = []  # biparite matrix二部矩阵
        if strGraph[0]=='bigraph':
            nodeNum = strGraph[2].split(',')
            nodeNum[0]=int(nodeNum[0])
            nodeNum[1]=int(nodeNum[1])
            strGraph[2] = nodeNum
            nodes = strGraph[3].split(',')
            nodes[0]=nodes[0].split(' ')
            nodes[1]=nodes[1].split(' ')
            strGraph[3] = nodes
            for i in range(nodeNum[0]):
                Mat.append([])
                for j in strGraph[i+4].split(' '):
                    Mat[-1].append(int(j))
        else:
            strGraph[2] = int(strGraph[2])
            strGraph[3] = strGraph[3].split(' ')
            for i in range(strGraph[2]):
                Mat.append([])
                for j in strGraph[i + 4].split(' '):
                    Mat[-1].append(int(j))
        print('解析点、边、矩阵:', strGraph)
        print('解析矩阵:',Mat)
        strGraph = strGraph[0:4]
        Mat = np.array(Mat)  # 把list转为numpy
        strGraph.append(Mat)
        self.global_strGraph = strGraph
        self.input_Mat = np.array(Mat,dtype=np.int)
        self.textBrowser.setText(str(''))
        self.label_2.setText("此处显示执行结果")
        print('整合:',self.global_strGraph)

    '''显示图到展示框'''
    def pushImg(self,des):
        '''
        从全局数据生成图，并push到指定展示框
        结果图是把原图中需要保留的路径设置为红色，不保留的设置黑色
        :param des: 目标展示框，origin or output
        :return:无
        '''
        #生成图文件
        strGraph=self.global_strGraph
        type=strGraph[0]
        weighted=strGraph[1]
        nodeNum=strGraph[2]
        nodesName=strGraph[3]
        print(strGraph)
        if des == 'input':
            Mat = self.input_Mat
        else:
            Mat = self.output_Mat
            self.textBrowser.setText(str(Mat))


        #根据图类型生成图类
        if type=='digraph': #有向图
            G=nx.DiGraph()
        else:               #无向图或二部图
            G=nx.Graph()

        #添加边和点
        colors = []
        if type=='bigraph': #二部图
            # 添加X集
            for n in nodesName[0]:
                print(n)
                G.add_node(n,desc=str(n),ipartite=0)
            # 添加Y集
            for n in nodesName[1]:
                print(n)
                G.add_node(n,desc=str(n),ipartite=1)
            # 添加匹配边

            for i in range(nodeNum[0]):  # 添加边
                for j in range(nodeNum[1]):
                    #输入还是输出？
                    #输入的图
                    if des=='input':
                        if self.input_Mat[i][j] == 0:  # 自身无环或非通路
                            G.add_edge(nodesName[0][i], nodesName[1][j])
                            colors.append('white')
                            continue
                        colors.append('black')
                        if weighted == 'weighted':
                            G.add_edge(nodesName[0][i], nodesName[1][j], name=Mat[i][j])
                        if weighted == 'unweighted':
                            G.add_edge(nodesName[0][i], nodesName[1][j])

                    if des=='output':
                        if (i,j) in self.M:
                            colors.append('red')
                        else:
                            colors.append('white')
                        if weighted == 'weighted':
                            G.add_edge(nodesName[0][i], nodesName[1][j], name=Mat[i][j])
                        if weighted == 'unweighted':
                            G.add_edge(nodesName[0][i], nodesName[1][j])


        else:               #无向图或有向图
            # 添加顶点
            for i in range(nodeNum):
                G.add_node(nodesName[i], desc=str(nodesName[i]))
            # 添加边
            for i in range(nodeNum):
                for j in range(nodeNum):
                    if Mat[i][j] == 0 or Mat[i][j] == -1:  # 自身无环或非通路
                        continue
                    colors.append('black')
                    G.add_edge(nodesName[i], nodesName[j], name=Mat[i][j])

        print('成功添加节点、边')
        cla()
        pos = nx.spring_layout(G)

        if type=='bigraph': #二部图要将两边的点垂直对齐
            print('????')
            l, r = nx.bipartite.sets(G)
            print(l,r)
            pos.update((node, (1, index)) for index, node in enumerate(l))
            pos.update((node, (2, index)) for index, node in enumerate(r))
        node_labels = nx.get_node_attributes(G, 'desc')
        nx.draw_networkx_labels(G, pos, labels=node_labels)
        edge_labels = nx.get_edge_attributes(G, 'name')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        nx.draw(G, pos , edge_color=colors)

        print('成功设置图样式')

        # 选择把图push到哪里
        if des=='input':    #显示原图
            imgPath=r'./process record/input.png'
        else:               #显示结果图
            imgPath=r'./process record/output.png'
        if not Path('./process record/').exists():
            os.mkdir(Path('./process record/'))

        # push图到对应位置
        plt.savefig(imgPath)
        sleep(2)            #睡眠防止读写顺序出错
        pix = QPixmap(imgPath).scaled(self.label.width(), self.label.height())
        if des=='input':    #显示原图
            self.label.setPixmap(pix)
        else:               #显示结果图
            self.label_2.setPixmap(pix)

    '''在输入区展示'''
    def showInput(self):
        self.getGraph()
        self.pushImg('input')

    '''构建UI界面'''
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "原图"))
        self.label.setText(_translate("MainWindow", "此处显示原图"))
        self.groupBox_2.setTitle(_translate("MainWindow", "结果图"))
        self.label_2.setText(_translate("MainWindow", "此处显示执行结果"))
        self.pushButton.setText(_translate("MainWindow", "获取图"))
        self.groupBox_3.setTitle(_translate("MainWindow", "最小生成树算法"))
        self.pushButton_2.setText(_translate("MainWindow", "破圈法"))
        self.pushButton_3.setText(_translate("MainWindow", "Kruskal"))
        self.pushButton_4.setText(_translate("MainWindow", "Prim"))
        self.groupBox_4.setTitle(_translate("MainWindow", "最大匹配算法"))
        self.pushButton_5.setText(_translate("MainWindow", "Hungarin"))
        self.groupBox_5.setTitle(_translate("MainWindow", "最短路径算法"))
        self.pushButton_8.setText(_translate("MainWindow", "Floyd"))
        self.pushButton_9.setText(_translate("MainWindow", "Dijkstra"))
        self.pushButton_6.setText(_translate("MainWindow", "Floyd–Warshall "))
        self.groupBox_6.setTitle(_translate("MainWindow", "输入邻接矩阵"))
        self.groupBox_7.setTitle(_translate("MainWindow", "输出邻接矩阵"))
        self.groupBox_8.setTitle(_translate("MainWindow", "最优匹配算法"))
        self.pushButton_7.setText(_translate("MainWindow", "Kuhn-Munkres"))