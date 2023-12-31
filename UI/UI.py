# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtGui
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem

import graph_model.graph as g

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1600, 900)
        Dialog.setMinimumSize(QtCore.QSize(1600, 900))
        Dialog.setMaximumSize(QtCore.QSize(1600, 900))
        self.listView = QtWidgets.QListView(Dialog)
        self.listView.setGeometry(QtCore.QRect(-50, -8, 1651, 911))
        self.listView.setStyleSheet("    background-image: url(./UI/background.jpg);\n"
"    background-position: center;\n"
"    background-repeat: no-repeat;\n"
"    background-attachment: fixed; \n"
"    background-size: contain;")
        self.listView.setObjectName("listView")
        self.pushButton_2 = QtWidgets.QPushButton(Dialog)
        self.pushButton_2.setGeometry(QtCore.QRect(50, 330, 111, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(Dialog)
        self.plainTextEdit.setGeometry(QtCore.QRect(12, 20, 191, 281))
        self.plainTextEdit.setOverwriteMode(False)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(160, 30, 41, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(860, 20, 711, 661))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.textBrowser = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser.setGeometry(QtCore.QRect(860, 690, 711, 191))
        self.textBrowser.setObjectName("textBrowser")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(220, 20, 631, 661))
        self.graphicsView.setObjectName("graphicsView")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(1460, 30, 101, 16))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(740, 30, 101, 16))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox_2 = QtWidgets.QComboBox(Dialog)
        self.comboBox_2.setGeometry(QtCore.QRect(0, 380, 211, 31))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(11)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(60, 460, 91, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setGeometry(QtCore.QRect(1510, 700, 41, 21))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.textBrowser_2 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_2.setGeometry(QtCore.QRect(0, 620, 211, 61))
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.textBrowser_3 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_3.setGeometry(QtCore.QRect(0, 690, 301, 211))
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.textBrowser_4 = QtWidgets.QTextBrowser(Dialog)
        self.textBrowser_4.setGeometry(QtCore.QRect(320, 690, 521, 211))
        self.textBrowser_4.setObjectName("textBrowser_4")

        self.retranslateUi(Dialog)
        self.pushButton_2.clicked.connect(self.getgraph)
        self.pushButton.clicked.connect(self.runGA)

    def getgraph(self):
            self.graph_model = g.Graph_Model(self.plainTextEdit)
            self.graph_model.generateImg('input')
            pixmap = QPixmap(r'./process record/input.png')
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # 创建场景和将图像项添加到场景
            scene = QtWidgets.QGraphicsScene()
            scene.addItem(pixmap_item)

            # 将场景设置到 graphicsView
            self.graphicsView.setScene(scene)

            # 如果需要，你可以调整视图大小以适应场景内容
            self.graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def runGA(self):
            current_text = self.comboBox_2.currentText()
            print(current_text)
            self.graph_model.RunGraphAlgorithm(current_text)
            self.graph_model.generateImg('output')
            pixmap = QPixmap(r'./process record/output.png')
            pixmap_item = QGraphicsPixmapItem(pixmap)
            # 创建场景和将图像项添加到场景
            scene = QtWidgets.QGraphicsScene()
            scene.addItem(pixmap_item)

            # 将场景设置到 graphicsView
            self.graphicsView_2.setScene(scene)

            # 如果需要，你可以调整视图大小以适应场景内容
            self.graphicsView_2.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)
            #
            if self.graph_model.output_Path is not None:
                    self.textBrowser.setText("path矩阵：\n" + str(self.graph_model.output_Path))
                    self.textBrowser.show()

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton_2.setText(_translate("Dialog", "显示图"))
        self.plainTextEdit.setPlainText(_translate("Dialog", "graph\n"
"weighted\n"
"6\n"
"A B C D E F\n"
"0 5 1 0 4 0\n"
"5 0 0 6 0 2\n"
"1 0 0 0 3 0\n"
"0 6 0 0 0 5\n"
"4 0 3 0 0 6\n"
"0 2 0 5 6 0"))
        self.plainTextEdit.setPlaceholderText(_translate("Dialog", "请按照要求输入"))
        self.label.setText(_translate("Dialog", "输入"))
        self.label_3.setText(_translate("Dialog", "结果可视化"))
        self.label_2.setText(_translate("Dialog", "图的可视化"))
        self.comboBox_2.setItemText(0, _translate("Dialog", "最小生成树Kruskal"))
        self.comboBox_2.setItemText(1, _translate("Dialog", "最小生成树Prime"))
        self.comboBox_2.setItemText(2, _translate("Dialog", "最小生成树破圈法"))
        self.comboBox_2.setItemText(3, _translate("Dialog", "最短路径Dijkstra"))
        self.comboBox_2.setItemText(4, _translate("Dialog", "最短路径Floyd"))
        self.comboBox_2.setItemText(5, _translate("Dialog", "最短路径Floyd-Warshall"))
        self.comboBox_2.setItemText(6, _translate("Dialog", "最大匹配匈牙利"))
        self.comboBox_2.setItemText(7, _translate("Dialog", "最优匹配Kuhn-Munkres"))
        self.pushButton.setText(_translate("Dialog", "确认"))
        self.label_5.setText(_translate("Dialog", "备注"))
        self.textBrowser_2.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:7.2pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; color:#0000ff;\">蒋宇、温思安、唐静媛、程佩</span></p>\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; color:#0000ff;\">2023年12月</span></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:7.2pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">提示</span><span style=\" font-size:11pt;\">:</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">请先根据“测试用例.txt”的格式</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">在输入框内输入邻接矩阵</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">然后点击“显示图”</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">再选择合适的算法</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">点击“确认”</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt;\">即可看到结果</span></p></body></html>"))
        self.textBrowser_4.setHtml(_translate("Dialog", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'SimSun\'; font-size:7.2pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:11pt; font-weight:600;\">输入格式:</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">第一行：graph or digraph or bigraph</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">第二行：weighted or unweighted</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">第三行：节点数：6 或者 二部图：5，6</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">第四行：节点名称：A B C 或者 x1 x2 x3,y1 y2 y3</span></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">第五行：键入邻接矩阵值</span></p></body></html>"))
# import background_rc
