import os
from time import sleep

import matplotlib
import networkx as nx
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.pyplot import cla
from algorithm import GraphAlgorithm as GA

matplotlib.use('TkAgg')


class Graph_Model:
    global_strGraph = None
    input_Mat = None
    output_Mat = None
    output_Path = None
    output_Msg = ""  # 字符串
    colors = None
    G = None
    seed_value = 42
    order_of_addition = []

    def __init__(self, plain_text_edit):
        try:
            self.output_Path = None
            graph_data = self.parse_graph_text(plain_text_edit.toPlainText())

            if graph_data:
                # TODO: 这是从文本框获取
                self.global_strGraph, self.input_Mat = self.process_graph_data(graph_data)
            else:
                self.global_strGraph = None
                self.input_Mat = None
        except Exception as e:
            print(f"Error in __init__: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志

    def parse_graph_text(self, text):
        try:
            if not text:
                return None
            graph_lines = text.split('\n')
            print("获取：", graph_lines)
            return graph_lines
        except Exception as e:
            print(f"Error in parse_graph_text: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志

    def process_graph_data(self, graph_lines):
        try:
            graph_type, weighted, node_counts, nodes_names, matrix = None, None, None, None, None

            if graph_lines[0] == 'bigraph':
                node_counts = list(map(int, graph_lines[2].split(',')))
                graph_lines[2] = node_counts

                nodes = [node.split(' ') for node in graph_lines[3].split(',')]
                graph_lines[3] = nodes

                matrix = [[int(j) for j in row.split(' ')] for row in graph_lines[4:]]
            else:  # 有向图和无向图
                graph_lines[2] = int(graph_lines[2])
                graph_lines[3] = graph_lines[3].split(' ')
                matrix = [[int(j) for j in row.split(' ')] for row in graph_lines[4:]]

            print('解析点、边、矩阵:', graph_lines)
            print('解析矩阵:', matrix)
            graph_data = graph_lines[:4]
            matrix_np = np.array(matrix)
            graph_data.append(matrix_np)

            print('整合:', graph_data)

            return graph_data, matrix_np
        except Exception as e:
            print(f"Error in parse_graph_data: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志

    def _create_graph(self, node_names, node_num, mat, des, weighted):
        try:
            G = nx.DiGraph() if self.global_strGraph[0] == 'digraph' else nx.Graph()
            colors = []
            # 针对二部图
            if self.global_strGraph[0] == 'bigraph':
                # 创建点
                for idx, nodes in enumerate(node_names):
                    bipartite_set = 0 if idx == 0 else 1
                    for n in nodes:
                        self.order_of_addition.append((n, bipartite_set))
                        G.add_node(n, desc=str(n), bipartite=bipartite_set)
                # 创建边
                for i in range(node_num[0]):
                    for j in range(node_num[1]):
                        if des == 'input':
                            if mat[i][j] == 0:
                                a = 0
                                # colors.append('white')
                                # G.add_edge(node_names[0][i], node_names[1][j], name="")
                            else:
                                colors.append('black')
                                if weighted == 'weighted':
                                    G.add_edge(node_names[0][i], node_names[1][j], name=str(mat[i][j]))
                                elif weighted == 'unweighted':
                                    G.add_edge(node_names[0][i], node_names[1][j], name="")
                        elif des == 'output':
                            if mat[i][j] != 0 and mat[i][j] != -1:
                                colors.append('red')
                                if weighted == 'weighted':
                                    G.add_edge(node_names[0][i], node_names[1][j])
                                elif weighted == 'unweighted':
                                    G.add_edge(node_names[0][i], node_names[1][j])
                            elif mat[i][j] == 0 and self.input_Mat[i][j] != 0:
                                colors.append('black')
                                if weighted == 'weighted':
                                    G.add_edge(node_names[0][i], node_names[1][j])
                                elif weighted == 'unweighted':
                                    G.add_edge(node_names[0][i], node_names[1][j])
            else:
                for i in range(node_num):
                    G.add_node(node_names[i], desc=str(node_names[i]))
                if self.global_strGraph[0] == 'graph':
                    transposed_matrix = np.transpose(mat)
                    # 取元素的逐元素最大值，得到无向图的邻接矩阵
                    mat = np.maximum(mat, transposed_matrix)
                    for i in range(node_num):
                        for j in range(node_num):
                            if des == 'input':
                                if mat[i][j] != 0 and not G.has_edge(node_names[i], node_names[j]):
                                    colors.append('black')
                                    G.add_edge(node_names[i], node_names[j], name=mat[i][j])
                            elif des == 'output':
                                if mat[i][j] != 0 and not G.has_edge(node_names[i], node_names[j]) and i!=j:
                                    colors.append('red')
                                    # print("涂红")
                                    G.add_edge(node_names[i], node_names[j], name=mat[i][j])
                                elif self.input_Mat[i][j] != 0  and not G.has_edge(
                                        node_names[i], node_names[j]) and i != j:
                                    colors.append('black')
                                    # print("涂黑")
                                    G.add_edge(node_names[i], node_names[j], name=self.input_Mat[i][j])
                if self.global_strGraph[0] == 'digraph':
                    for i in range(node_num):
                        for j in range(node_num):
                            if des == 'input':
                                if mat[i][j] != 0 and i!=j :
                                    colors.append('black')
                                    G.add_edge(node_names[i], node_names[j], name=mat[i][j])
                            elif des == 'output':
                                if mat[i][j] != 0 and i!=j:
                                    colors.append('red')
                                    G.add_edge(node_names[i], node_names[j], name=mat[i][j])
                                elif self.input_Mat[i][j] != 0 and self.input_Mat[i][j] != -1:
                                    colors.append('black')
                                    G.add_edge(node_names[i], node_names[j], name=self.input_Mat[i][j])
            return G, colors
        except Exception as e:
            print(f"Error in create_graph: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志

    def _draw_graph(self, G, pos, colors, type_graph):
        try:
            cla()
            if type_graph == 'bigraph':
                print('二部图')
                left, right = [], []
                for node, bipartite_set in self.order_of_addition:
                    if bipartite_set == 0:
                        left.append(node)
                    else:
                        right.append(node)
                print(left, right)
                pos.update((node, (1, index)) for index, node in enumerate(left))
                pos.update((node, (2, index)) for index, node in enumerate(right))
            nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'desc'))
            edge_labels = nx.get_edge_attributes(G, 'name')
            # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.3, font_color="blue", font_size=15, rotate=False,
            #                              bbox=dict(boxstyle="round,pad=0.3",edgecolor="yellow",facecolor="yellow"))
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, label_pos=0.4, font_color="blue",
                                         font_size=15, rotate=False
                                         )
            nx.draw(G, pos, edge_color=colors)
        except Exception as e:
            print(f"Error in _draw_graph: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志

    def _save_graph_image(self, img_path):
        plt.savefig(img_path)
        sleep(1)  # Sleep to prevent read/write order issues

    def generateImg(self, des):
        """
        从模型中的数据生成图
        结果图是把原图中需要保留的路径设置为红色，不保留的设置黑色
        :param des: 输入还是输出input or output
        :return:无
        """
        try:
            # Check if essential data is available
            if self.global_strGraph is None or self.input_Mat is None:
                raise ValueError("Graph data is missing.")
            # Extract data from the global graph representation
            type_graph, weighted, node_num, node_names, mat = self.global_strGraph
            mat = np.array(mat)  # Convert list to NumPy array
            if des == 'input':
                mat = self.input_Mat
            elif des == 'output' and (self.output_Mat is None):
                raise ValueError("Output matrix is missing or empty.")
            else:
                mat = self.output_Mat

            # Create and draw the graph 此处修改，记得colors并不应该是全局变量，到时候做个判断改回去
            self.G = None
            self.colors = None
            self.G, self.colors = self._create_graph(node_names, node_num, mat, des, weighted)
            # print(self.G.edges)
            # print(self.colors)
            # print(mat)
            pos = nx.spring_layout(self.G, seed=self.seed_value)
            self._draw_graph(self.G, pos, self.colors, type_graph)
            # 这里看颜色里有啥
            # Save the graph image
            img_path = r'./process record/input.png' if des == 'input' else r'./process record/output.png'
            if not Path('../process record/').exists():
                os.mkdir(Path('../process record/'))
            self._save_graph_image(img_path)

        except Exception as e:
            print(f"Error in generateImg: {e}")
            # You can add additional error handling here, such as displaying an error message

    def RunGraphAlgorithm(self, algorithm_name: str):
        try:
            """
            根据传入的算法名称执行相应的算法。
    
            Parameters:
            - algorithm_name (str): 要运行的算法名称。
    
            Returns:
            - np.ndarray: NumPy矩阵作为算法的结果。
            """
            self.output_Path = ""
            if algorithm_name == "最小生成树Kruskal":
                """
                记住这里需要删除，测试用！！！！
                """
                self.output_Mat = GA.runKruskal(self.input_Mat)
            elif algorithm_name == "最小生成树Prime":
                self.output_Mat = GA.runPrim(self.input_Mat)
            elif algorithm_name == "最小生成树破圈法":
                self.output_Mat = GA.runDes_Cir(self.input_Mat)
            elif algorithm_name == "最短路径Dijkstra":
                self.output_Mat,self.output_Path = GA.runDijkstra(self.input_Mat)
                self.output_Msg = "Path矩阵：\n" + str(self.output_Path)

            elif algorithm_name == "最短路径Floyd":
                self.output_Mat, self.output_Path = GA.runFloyd(self.input_Mat)
                self.output_Msg = "Path矩阵：\n" + str(self.output_Path)
                for i in range(self.output_Mat.shape[0]):
                    if self.output_Mat[i][i] < 0:
                        self.output_Msg += "\nWarning: This Input Graph Have Negative Circle!\n"
                        raise Exception("This Input Graph Have Negative Circle!")

            elif algorithm_name == "最短路径Floyd-Warshall":
                self.output_Mat, self.output_Path = GA.runFloydwarshall(self.input_Mat)
                self.output_Msg = "Path矩阵：\n" + str(self.output_Path)
                for i in range(self.output_Mat.shape[0]):
                    if self.output_Mat[i][i] < 0:
                        self.output_Msg += "\nWarning: This Input Graph Have Negative Circle!\n"  # 备注输出框 加上 显示警告
                        raise Exception("This Input Graph Have Negative Circle!")

            elif algorithm_name == "最大匹配匈牙利":
                self.output_Mat = GA.runHungarian(self.input_Mat)
            elif algorithm_name == "最优匹配Kuhn-Munkres":
                self.output_Mat = GA.runKuhn_Munkres(self.input_Mat)
        except Exception as e:
            print(f"Error in RunGraphAlgorithm: {e}")
            # 添加错误处理，例如显示错误消息或者记录错误日志
