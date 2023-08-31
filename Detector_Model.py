# -*- coding: utf-8 -*-
# 本程序用于视频中车辆行人等多目标检测跟踪
# @Time    : 2021/3/13 10:29
# @Author  : sixuwuxian
# @Email   : sixuwuxian@aliyun.com
# @blog    : wuxian.blog.csdn.net
# @Software: PyCharm

import os
import time

import cv2
import numpy as np

from sort import Sort


class Detector:
    def __init__(self, model_path=None):
        CAM_NUM = 0  # 摄像头序号
        if_save = 1  # 是否需要保存录制的视频，1表示保存
        self.filter_confidence = 0.5  # 用于筛除置信度过低的识别结果
        self.threshold_prob = 0.3  # 用于NMS去除重复的锚框
        self.tracker = Sort()#创建实例对象
        if model_path is None:
            model_path = "D:\W\沫姐项目\pyqt\qiao\weights"  # 模型文件的目录
        # 载入模型参数文件及配置文件
        weightsPath = os.path.sep.join([model_path, "best.pt"])
        configPath = os.path.sep.join([model_path, "voc.yaml"])
        # 载入数据集标签
        # labelsPath = os.path.sep.join(["/home/hcb/mojie_project/AAAAAA/yolo-obj", "yolo3_object.names"])
        labelsPath = os.path.sep.join(["D:\W\沫姐项目\pyqt\qiao\weights", "coco.names"])
        self.LABELS = open(labelsPath).read().strip().split("\n")
        # 从配置和参数文件中载入模型

        self.net = cv2.dnn_DetectionModel(configPath, weightsPath)
        # self.net = cv2.dnn.readNet(weightsPath,configPath)
        ln = self.net.getLayerNames()
        self.ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]# #得到未连接层得序号
        """yolo的输出层是未连接层的前一个元素，通过net.getUnconnectedOutLayers()找到未连接层的序号out= [[200] /n [267] /n [400] ]，循环找到所有的输出层，赋值给ln
最终ln = [‘yolo_82’, ‘yolo_94’, ‘yolo_106’"""
        # except:
        #     print("读取模型失败，请检查文件路径并确保无中文文件夹！")

    def run(self, frame):
        frame_in = frame.copy()
        # 将一帧画面读入网络
        blob = cv2.dnn.blobFromImage(frame_in, 1 / 255.0, (416, 416), swapRB=True, crop=False)#支持调用的深度学习框架，主要作用是对图像进行预处理
        self.net.setInput(blob)

        start = time.time()
        layerOutputs = self.net.forward(self.ln)#前向传播
        end = time.time()

        boxes = []  # 用于检测框坐标
        confidences = []  # 用于存放置信度值# 表示识别目标是某种物体的可信度
        classIDs = []  # 用于识别的类别序号# 表示识别的目标归属于哪一类，['person', 'bicycle', 'car', 'motorbike'....]

        (H, W) = frame_in.shape[:2]

        # 逐层遍历网络获取输出
        for output in layerOutputs:
            #遍历每一次检测
            for detection in output:
                # #提取当前对象检测的类ID和置信度(即概率)
                scores = detection[5:]#detection里有[x,y,w,h,conf,score1,socre2,...score80]# 当前目标属于某一类别的概率
                classID = np.argmax(scores)#返回的类别分数最大的索引值
                confidence = scores[classID]# 得到目标属于该类别的置信度

                # 过滤低置信度值的检测结果
                if confidence > self.filter_confidence:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # 转换标记框
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # 更新标记框、置信度值、类别列表
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # 使用NMS去除重复的标记框
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.filter_confidence, self.threshold_prob)#NMSBoxes()函数返回值为最终剩下的按置信度由高到低的矩形框的序列号

        dets = [] # 存放检测框的信息，包括左上角横坐标，纵坐标，右下角横坐标，纵坐标，以及检测到的物体的置信度和物体类别用于目标跟踪
        if len(idxs) > 0:
            # 遍历索引得到检测结果
            for i in idxs.flatten():#一维数组#循环检测出的每一个box
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                dets.append([x, y, x + w, y + h, confidences[i], classIDs[i]])# 将检测框的信息的放入dets中

        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})#用于控制Python中小数的显示精度。
        dets = np.asarray(dets)

        # 使用sort算法，开始进行追踪
        tracks = self.tracker.update(dets)## 将检测结果传入跟踪器中，返回当前画面中跟踪成功的目标，包含五个信息：目标框的左上角和右下角横纵坐标，目标的置信度
        boxes = []  # 存放tracks中的前四个值：目标框的左上角横纵坐标和右下角的横纵坐标
        indexIDs = []  #  存放追踪到的序号
        cls_IDs = [] # 存放追踪到的类别

        for track in tracks:
            boxes.append([track[0], track[1], track[2], track[3]])# 更新目标框坐标信息
            indexIDs.append(int(track[4]))# 存放追踪到的序号
            cls_IDs.append(int(track[5]))# # 存放追踪到的类别

        return dets, boxes, indexIDs, cls_IDs
