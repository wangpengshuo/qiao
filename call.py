import json
import os
import time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMessageBox

import rtsp_camera
from interface import Ui_MainWindow
import serial
import serial.tools.list_ports
import sys
import cv2
import threading
from utils.capnums import Camera
from DetThread import DetThread
from dialog.rtsp_win import Window
from utils.CustomMessageBox import MessageBox
from rtsp_camera import rtsp_camera

class MainForm(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainForm, self).__init__()
        self.setupUi(self)
        # self.startBn.clicked.connect(self.robot_start)
        # self.stopBn.clicked.connect(self.robot_stop)
        self.openLightBn.clicked.connect(self.robot_openLight)
        self.closeLightBn.clicked.connect(self.robot_closeLight)
        # self.clearBn.clicked.connect(self.arm_clear)
        # self.sendBn.clicked.connect(self.arm_send)
        # self.detection_openBn.clicked.connect(self.detection_open)
        # self.detection_snapBn.clicked.connect(self.detection_snap)
        # self.detection_clearBn.clicked.connect(self.detection_clear)
        ########摄像头模块#######
        # self.timer_camera = QtCore.QTimer()  # 定时器
        # self.CAM_NUM = 0  # 摄像头标号
        # self.cap = cv2.VideoCapture(self.CAM_NUM)  # 屏幕画面对象
        # self.cap_video = None  # 视频流对象
        # self.current_image = None  # 保存的画面
        # self.detected_image = None
        #
        # self.timer_camera.timeout.connect(self.show_camera)
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
        # self.ReceNumForClear = 0  # 接收框的数据个数用于自动清除  放到初始化只会初始化一次
        ###########开始即检测###
        # 定时清空自定义状态栏上的文字
        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # 自动搜索模型
        self.comboBox.clear()
        self.pt_list = os.listdir('./weights')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5线程
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type  # 权重
        self.det_thread.source = "0"  # 默认打开本机摄像头，无需保存到配置文件
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.label_display))
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))
        self.cameraButton.clicked.connect(self.chose_cam)
        self.runButton.clicked.connect(self.run_or_continue)
        self.rtspButton.clicked.connect(self.chose_rtsp)
        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox_3.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider_3.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox_3.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider_3.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox_3.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider_3.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.load_setting()

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            new_config = {"iou": 0.26,
                          "conf": 0.33,
                          "rate": 10,
                          "check": 0
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            iou = config['iou']
            conf = config['conf']
            rate = config['rate']
        self.confSpinBox_3.setValue(iou)
        self.iouSpinBox_3.setValue(conf)
        self.rateSpinBox_3.setValue(rate)

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider_3.setValue(int(x * 100))
        elif flag == 'confSlider':
            self.confSpinBox_3.setValue(x / 100)
            self.det_thread.conf_thres = x / 100
        elif flag == 'iouSpinBox':
            self.iouSlider_3.setValue(int(x * 100))
        elif flag == 'iouSlider':
            self.iouSpinBox_3.setValue(x / 100)
            self.det_thread.iou_thres = x / 100
        elif flag == 'rateSpinBox':
            self.rateSlider_3.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox_3.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def chose_rtsp(self):
        # threading.Thread(target=self.load_rtsp('rtsp://192.168.43.177:8554/ds-test', self.label_display)).start()
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://192.168.43.177:8554/live1.h264"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))
        # print("pid:", os.getpid())
        # self.rtsp_camera = rtsp_camera('rtsp://192.168.43.177:8554/live1.h264',self.label_display)
        # self.rtsp_window.rtspButton.clicked.connect(
        #     threading.Thread(target=self.rtsp_camera.display).start())

    def stop(self):
        self.det_thread.jump_out = True

    def load_rtsp(self, url):
        # cap = cv2.VideoCapture(url)
        # start_time = time.time()
        # while cap.isOpened():
        #     success, frame = cap.read()
        #     if success:
        #         if (time.time() - start_time) > 0.1:
        #             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #             img = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        #             label.setPixmap(QPixmap.fromImage(img))
        #             cv2.waitKey(1)
        #             start_time = time.time()
        try:
            self.stop()
            self.det_thread.source = url
            new_config = {"ip": url}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('加载rtsp：{}'.format(url))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            # MessageBox的作用：留出2秒，让上一次摄像头安全release
            # 自动检测本机有哪些摄像头
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('加载摄像头：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = '摄像头设备' if source.isnumeric() else source
            self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('暂停')

    def search_pt(self):
        pt_list = os.listdir('./weights')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./weights/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./weights/%s" % self.model_type
        self.statistic_msg('模型切换为%s' % x)

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # 保持纵横比
            # 找出长边
            if iw > ih:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_RGB2BGR)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)   # 3秒后自动清除
        print(msg)

    def port_check(self):
        # 检测所有存在的串口，将信息存储在字典中
        self.Com_Dict = {}  # 创建一个字典，字典是可变的容器
        port_list = list(serial.tools.list_ports.comports())  # list是序列，一串数据，可以追加数据
        for port in port_list:
            self.Com_Dict["%s" % port[0]] = "%s" % port[1]
        if len(self.Com_Dict) == 0:
            print(" 无串口")

        self.ser.port = 'COM7'  # 串口选择框
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
            # self.port_close()
            return None

        if num > 0:  # 如果收到数据 数据以十六进制的形式放到data里面     ,接收一次数据进一次（也就是发送端点一次发送进一次）
            data = self.ser.read(num)  # 从串口读取指定字节大小的数据
            D = data.decode('iso-8859-1')
            D = D.strip()
            # self.distanceLb.setText(data.decode('iso-8859-1'))  # 接收区
            self.distance.setText(D)
            self.det_thread.length = D

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
        pass
        # self.det_thread.jump_out = False
        # if self.detection_openBn.isChecked():
        #     self.det_thread.is_continue = True
        #     if not self.det_thread.isRunning():
        #         self.det_thread.start()
        #     source = os.path.basename(self.det_thread.source)
        #     source = '摄像头设备' if source.isnumeric() else source
        #     self.statistic_msg('正在检测 >> 模型：{}，文件：{}'.
        #                        format(os.path.basename(self.det_thread.weights),
        #                               source))
        # else:
        #     self.det_thread.is_continue = False
        #     self.statistic_msg('暂停')
        # if not self.timer_camera.isActive():  # 检查定时状态
        #     flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
        #     if not flag:  # 相机打开失败提示
        #         msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
        #                                             u"请检测相机与电脑是否连接正确！ ",
        #                                             buttons=QtWidgets.QMessageBox.Ok,
        #                                             defaultButton=QtWidgets.QMessageBox.Ok)
        #         self.flag_timer = ""
        #     else:
        #         # 准备运行识别程序
        #         self.flag_timer = "camera"
        #         self.timer_camera.start(30)
        # else:
        #     # 定时器未开启，界面回复初始状态
        #     self.flag_timer = ""
        #     self.timer_camera.stop()

    # def show_camera(self):
    #     flag, image = self.cap.read()  # 获取画面
    #     if flag:
    #         self.c = 1
    #         # image = cv2.flip(image, 1)  # 左右翻转
    #         # 在Qt界面中显示检测完成画面
    #         image = cv2.resize(image, (1152, 648))  # 设定图像尺寸为显示界面大小
    #         show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
    #         self.label_display.setPixmap(QtGui.QPixmap.fromImage(showImage))
    #     else:
    #         self.timer_camera.stop()
    #
    # def find_marker(self, image):
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 颜色空间转换函数，RGB和BGR颜色空间转换 opencv默认的颜色空间是BGR
    #     gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波，对图像进行滤波操作 ，（5,5）表示高斯核的大小 ，0 表示标准差取0
    #     edged = cv2.Canny(gray, 50, 150)  # canny 算子 边缘检测 35是阈值1， 125是阈值2，大的阈值用于检测图像中的明显边缘，小的阈值用于将不明显的边缘检测连接起来
    #
    #     (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST,
    #                                  cv2.CHAIN_APPROX_SIMPLE)  # 找到详细的轮廓点， RETR_LIST 以列表的形式输出轮廓信息
    #     # CHAIN_APPROX_SIMPLE： 压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标
    #     c = max(cnts, key=cv2.contourArea)  # 轮廓点的面积计算
    #     # return edged
    #     return cv2.minAreaRect(c)  # 求出在 C点集下的像素点的面积
    #
    # def object_width(self, D, P, F):
    #     return (D * P) / F

    # def detection_snap(self):
    #     flag, image = self.cap.read()  # 获取画面
    #     num = 0
    #     if flag:
    #         cv2.imwrite('./image/' + str(num) + '.jpg', image)
    #     print(self.distance.text())
    #
    #     D = self.distance.text()  # 需要手动测量目标的宽度，单位为cm
    #     D = float(D)
    #     self.F = 19.98808078471807  # 根据get_F求出 ,get_F()函数是为了求得相机的焦距，需要通过测试图像中的目标距离来求出
    #     self.frame = cv2.imread('./image/0.jpg')
    #     marker = self.find_marker(self.frame)
    #     self.P = marker[1][0] / 18.95  # 300dim 1cm = 118.11像素值 ，300dim指300分辨率，有1080分辨率，像素值的㎝转换是不同
    #     self.inches = round(self.object_width(D, self.P, self.F), 2)  #
    #     self.inches = round(self.inches + 10 - 10, 2)
    #     # draw a bounding box around the image and display it
    #     box = cv2.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    #     box = np.int0(box)
    #     cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
    #     cv2.putText(image, "%.2fcm" % self.inches,
    #                 (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
    #                 2.0, (0, 255, 0), 3)
    #     cv2.imwrite('./image/' + '1.jpg', image)
    #
    #     self.length.setText(str(self.inches))
    #     self.cap.release()
    #     self.label_display.setPixmap(QPixmap("./image/1.jpg"))

    # def detection_clear(self):
    #     if not self.timer_camera.isActive():  # 检查定时状态
    #         flag = self.cap.open(self.CAM_NUM)  # 检查相机状态
    #         if not flag:  # 相机打开失败提示
    #             msg = QtWidgets.QMessageBox.warning(self.centralwidget, u"Warning",
    #                                                 u"请检测相机与电脑是否连接正确！ ",
    #                                                 buttons=QtWidgets.QMessageBox.Ok,
    #                                                 defaultButton=QtWidgets.QMessageBox.Ok)
    #             self.flag_timer = ""
    #         else:
    #             # 准备运行识别程序
    #             self.flag_timer = "camera"
    #             self.timer_camera.start(30)
    #     else:
    #         # 定时器未开启，界面回复初始状态
    #         self.flag_timer = ""
    #         self.timer_camera.stop()
    #         # self.length.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywin = MainForm()
    mywin.show()
    sys.exit(app.exec_())
