import json
import os
import time

from PyQt5 import QtWidgets, QtGui, QtSql
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
from interface import Ui_MainWindow
import serial
import serial.tools.list_ports
import torch
import torch.backends.cudnn as cudnn
import sys
import cv2
import math
import imutils
import re
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device, load_classifier, time_sync
from utils.capnums import Camera
from queue import LifoQueue

Decode2Play = LifoQueue()

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # 发送信号：正在检测/暂停/停止/检测结束/错误报告
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = '.\weights\YOLOv5s.pt'  # 设置权重
        self.current_weight = '.\weights\YOLOv5s.pt'  # 当前权重
        self.source = '0'  # 视频源
        self.conf_thres = 0.25  # 置信度
        self.iou_thres = 0.45  # iou
        self.jump_out = False  # 跳出循环
        self.is_continue = True  # 继续/暂停
        self.percent_length = 1000  # 进度条
        self.rate_check = False  # 是否启用延时
        self.rate = 1  # 延时HZ
        self.length=60

        self.threadFlag = 0  # 控制线程退出
        self.changeFlag = 0
        self.threadFlag = 0  # 控制线程退出
        self.playFlag = 0
        self.playLabel = QLabel()  # 初始化QLabel对象
    @torch.no_grad()
    def run(self,
            imgsz=640,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=3,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=False,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):
        try:
            device = select_device(device)
            half &= device.type != 'cpu'
            print(self.length)

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(
                    ('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            # 跳帧检测
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)
            while True:
                # 手动停止
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_msg.emit('停止')
                    break
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    # 每三十帧刷新一次输出帧率
                    if count % 30 == 0 and count >= 30:
                        fps = int(30 / (time.time() - start_time))
                        self.send_fps.emit('fps：' + str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.percent_length)
                        self.send_percent.emit(percent)
                        time.sleep(0.001)
                        Decode2Play.put(img)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms,
                                               max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        im0 = im0s.copy()

                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                # im0 = plot_one_box_PIL(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)  # 中文标签画框，但是耗时会增加
                                plot_one_box(xyxy, im0, label=label, color=colors(c, True),
                                             line_thickness=line_thickness, length=self.length)


                    # 控制视频发送频率
                    if self.rate_check:
                        time.sleep(1 / self.rate)
                    # print(type(im0s))
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if percent == self.percent_length:
                        self.send_percent.emit(0)
                        self.send_msg.emit('检测结束')
                        # 正常跳出循环
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)
