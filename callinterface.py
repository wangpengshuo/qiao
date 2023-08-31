import os
import time
from collections import deque
import json
import serial
import serial.tools.list_ports
import torch
import torch.backends.cudnn as cudnn
from PyQt5 import QtWidgets, QtGui, QtSql
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import QMessageBox, QMainWindow, QApplication
from PyQt5 import QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox
import sys
import cv2
import math
import imutils
import re
import numpy as np
from interface import Ui_MainWindow

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.CustomMessageBox import MessageBox
# LoadWebcam 的最后一个返回值改为 self.cap
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import Colors, plot_one_box, plot_one_box_PIL
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.capnums import Camera


# from dialog.rtsp_win import Window #推流
class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        self.startBn.clicked.connect(self.robot_start)
        self.stopBn.clicked.connect(self.robot_stop)
        self.openLightBn.clicked.connect(self.robot_openLight)
        self.closeLightBn.clicked.connect(self.robot_closeLight)
        self.clearBn.clicked.connect(self.arm_clear)
        self.sendBn.clicked.connect(self.arm_send)
        self.detection_openBn.clicked.connect(self.detection_open)
        self.detection_snapBn.clicked.connect(self.detection_snap)
        self.detection_clearBn.clicked.connect(self.detection_clear)
        ########摄像头模块#######
        self.timer_camera = QtCore.QTimer()  # 定时器
        self.CAM_NUM = 0  # 摄像头标号
        self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象
        self.cap_video = None  # 视频流对象
        self.current_image = None  # 保存的画面
        self.detected_image = None

        self.timer_camera.timeout.connect(self.show_camera)
        #######串口助手########
        self.timer_open = QTimer()
        self.timer_open.timeout.connect(self.robot_openLight)
        self.timer_close = QTimer()
        self.timer_close.timeout.connect(self.robot_closeLight)
        self.tit = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.data_receive)  # 串口中断是2ms进一次
        self.ser = serial.Serial()
        self.port_check()
        self.data_num_received = 0  # 接收数据清零
        self.data_num_sended = 0  # 发送数据清零
        self.ReceNumForClear = 0  # 接收框的数据个数用于自动清除  放到初始化只会初始化一次

    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}  # 创建一个字典，字典是可变的容器
        port_list = list(serial.tools.list_ports.comports())  # list是序列，一串数据，可以追加数据
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
        if len(self.Com_Dict) == 0:
            print(" 无串口")

        self.ser.port = 'COM10'  # 串口选择框
        self.ser.baudrate = 115200  # 波特率输入框
        self.ser.bytesize = 8  # 数据位输入框
        self.ser.stopbits = 1  # 停止位输入框
        self.ser.parity = 'N'  # 校验位输入框
        try:
            self.ser.open()
        except:
            QMessageBox.critical(self, "Port Error", "此串口不能被打开！")
            return None
        self.timer.start(2)  # 打开串口接收定时器，周期为2ms

    def port_close(self):
        self.timer.stop()  # 停止计时器
        self.timer_send.stop()  # 停止定时发送
        try:
            self.ser.close()
        except:
            pass
        # 接收数据和发送数据数目置零
        self.data_num_received = 0
        self.data_num_sended = 0

    # ----------------------接收数据
    def data_receive(self):
        try:
            num = self.ser.inWaiting()  # 获取接受缓存中的字符数
        except:
            self.port_close()
            return None

        if num > 0:  # 如果收到数据 数据以十六进制的形式放到data里面     ,接收一次数据进一次（也就是发送端点一次发送进一次）
            data = self.ser.read(num)  # 从串口读取指定字节大小的数据

            # self.distanceLb.setText(data.decode('iso-8859-1'))  # 接收区
            self.distance.setText(data.decode('iso-8859-1'))
        else:
            pass

    def robot_start(self):
        pass

    def robot_stop(self):
        pass

    def robot_openLight(self):
        if self.ser.isOpen():  # 判断串口是否打开 ↓
            input_s = '1'  # （发送区）获取文本内容
            input_s = (input_s.encode('utf-8'))  #
            num = self.ser.write(input_s)  # 串口写，返回的写入的数据数
            self.data_num_sended += num  # 发送数据统计

            print(input_s)

        else:
            pass  # 空语句 保证结构的完整性

    def robot_closeLight(self):
        if self.ser.isOpen():  # 判断串口是否打开 ↓
            input_s = '0'  # （发送区）获取文本内容
            input_s = (input_s.encode('utf-8'))  #
            num = self.ser.write(input_s)  # 串口写，返回的写入的数据数
            self.data_num_sended += num  # 发送数据统计
            print(input_s)

    def arm_clear(self):
        pass

    def arm_send(self):
        pass

    def detection_open(self):
        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                # 准备运行识别程序
                self.det_thread.is_continue = True
                self.flag_timer = "camera"
                self.label_display.setText('正在启动识别系统...\n\nloading')
                # 打开定时器
                self.timer_camera.start(30)
        else:
            # 定时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()

    def show_camera(self):
        # self.time_count =0
        flag, image = self.cap.read()  # 获取画面
        if flag:
            image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
            show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))
        else:
            self.timer_video.stop()

    def find_marker(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色空间转换函数，RGB和BGR颜色空间转换 opencv默认的颜色空间是BGR
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波，对图像进行滤波操作 ，（5,5）表示高斯核的大小 ，0 表示标准差取0
        edged = cv2.Canny(gray, 50, 150)  # canny 算子 边缘检测 35是阈值1， 125是阈值2，大的阈值用于检测图像中的明显边缘，小的阈值用于将不明显的边缘检测连接起来

        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_SIMPLE)  # 找到详细的轮廓点， RETR_LIST 以列表的形式输出轮廓信息
        # CHAIN_APPROX_SIMPLE： 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
        c = max(cnts, key=cv2.contourArea)  # 轮廓点的面积计算
        # return edged
        return cv2.minAreaRect(c)  # 求出在 C点集下的像素点的面积

    def object_width(self, D, P, F):
        return (D * P) / F

    def detection_snap(self):
        flag, image = self.cap.read()  # 获取画面
        num = 0
        if flag:
            cv2.imwrite('./image/' + str(num) + '.jpg', image)
        print(self.distance.text())

        D = self.distance.text()  # 需要手动测量目标的宽度，单位为cm
        D = float(D)
        self.F = 19.98808078471807  # 根据get_F求出 ,get_F()函数是为了求得相机的焦距，需要通过测试图像中的目标距离来求出
        self.frame = cv2.imread('./image/0.jpg')
        marker = self.find_marker(self.frame)
        self.P = marker[1][0] / 18.95  # 300dim 1cm = 118.11像素值 ，300dim指300分辨率，有1080分辨率，像素值的㎝转换是不同
        self.inches = round(self.object_width(D, self.P, self.F), 2)  #
        self.inches = round(self.inches + 10 - 10, 2)
        # draw a bounding box around the image and display it
        box = cv2.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
        box = np.int0(box)
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        cv2.putText(image, "%.2fcm" % self.inches,
                    (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 3)
        cv2.imwrite('./image/' + '1.jpg', image)

        self.length.setText(str(self.inches))
        self.cap.release()
        self.label_display.setPixmap(QPixmap("./image/1.jpg"))

    def detection_clear(self):
        if not self.timer_camera.isActive():  # 检查定时状态
            flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
            if not flag:  # 相机打开失败提示
                msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
                                                    u"请检测相机与电脑是否连接正确！ ",
                                                    buttons=QtWidgets.QMessageBox.Ok,
                                                    defaultButton=QtWidgets.QMessageBox.Ok)
                self.flag_timer = ""
            else:
                # 准备运行识别程序
                self.flag_timer = "camera"
                self.timer_camera.start(30)
        else:
            # 定 时器未开启，界面回复初始状态
            self.flag_timer = ""
            self.timer_camera.stop()
            # self.length.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywin = MainForm()
    mywin.show()
    sys.exit(app.exec_())
