# 作者：gc
# 日期：2020-04-09 14:06
# 描述：测试主类
import cv2, sys, os, threading, time
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap

class rtsp_camera:
    """摄像头对象"""

    def __init__(self, url, out_label):
        """初始化方法"""
        self.url = url
        self.outLabel = out_label

    def display(self):
        """显示"""
        cap = cv2.VideoCapture(self.url)
        start_time = time.time()
        while cap.isOpened():
            success, frame = cap.read()
            if success:
                if (time.time() - start_time) > 0.1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
                    self.outLabel.setPixmap(QPixmap.fromImage(img))
                    cv2.waitKey(1)
                    start_time = time.time()


