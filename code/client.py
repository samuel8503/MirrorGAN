from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QThread
from PyQt5.QtCore import QTimer
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from paramiko import SSHClient
from scp import SCPClient

import ui2
import sys
import socket
import pickle

class SubmitThread(QThread):
    def __init__(self, main, data):
        QThread.__init__(self)
        self.data = data
        self.main = main

    def run(self):
        try:
            s = socket.socket()
            s.connect(self.main.addr)
        except socket.timeout:
            print("ERROR: Request timeout.")
            if self.data['dataset'] == "bird":
                self.main.bird_show_sent.setText("ERROR: Request timeout")
                self.main.bird_show_sent.setStyleSheet("QLabel { color : red; }")
            else:
                self.main.cooc_show_sent.setText("ERROR: Request timeout")
                self.main.coco_show_sent.setStyleSheet("QLabel { color : red; }")
            s.close()
            return
        s.sendall(pickle.dumps(self.data))
        result = pickle.loads(s.recv(4096))
        if result['success']:
            with SCPClient(self.main.ssh.get_transport()) as scp:
                if self.data['dataset'] == "bird":
                    scp.get('~/MirrorGAN/models/bird_600/network_ipc', recursive=True)
                    self.main.bird_g0.setPixmap(QPixmap("network_ipc/0_s_0_g0.png"))
                    self.main.bird_g1.setPixmap(QPixmap("network_ipc/0_s_0_g1.png"))
                    self.main.bird_g2.setPixmap(QPixmap("network_ipc/0_s_0_g2.png"))
                    self.main.bird_a1.setPixmap(QPixmap("network_ipc/0_s_0_a1.png"))
                    self.main.bird_show_sent.setText("Input: " + self.data['str'])
                    self.main.bird_show_sent.setStyleSheet("QLabel { color : green; }")
                else:
                    scp.get('~/MirrorGAN/output/coco_glu-gan2_2019_07_14_13_06_53/Model/netG_epoch_36/network_ipc', recursive=True)
                    self.main.coco_g0.setPixmap(QPixmap("network_ipc/0_s_0_g0.png"))
                    self.main.coco_g1.setPixmap(QPixmap("network_ipc/0_s_0_g1.png"))
                    self.main.coco_g2.setPixmap(QPixmap("network_ipc/0_s_0_g2.png"))
                    self.main.coco_a1.setPixmap(QPixmap("network_ipc/0_s_0_a1.png"))
                    self.main.coco_show_sent.setText("Input: " + self.data['str'])
                    self.main.coco_show_sent.setStyleSheet("QLabel { color : green; }")
        else:
            print(result['msg'])
            if self.data['dataset'] == "bird":
                self.main.bird_show_sent.setText("ERROR: Error occurred, please check log")
                self.main.bird_show_sent.setStyleSheet("QLabel { color : red; }")
            else:
                self.main.coco_show_sent.setText("ERROR: Error occurred, please check log")
                self.main.coco_show_sent.setStyleSheet("QLabel { color : red; }")
        s.close()

    def stop(self):
        self.threadactive = False
        self.wait()

class Dialog(QDialog, ui2.Ui_Dialog):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.setWindowModality(Qt.ApplicationModal)

class Main(QMainWindow, ui2.Ui_MainWindow):
    def __init__(self):
        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.addr = ("140.113.207.102", 59488)
        self.ssh = SSHClient()
        self.ssh.load_system_host_keys()
        self.ssh.connect(self.addr[0], username="samuel", password="jerryy.3d9 x87")
        self.dialog = Dialog()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dialog)

    def bird_submit(self):
        data = {'dataset': 'bird'}
        data['str'] = self.bird_input.text()
        self.worker = SubmitThread(self, data)
        self.worker.finished.connect(self.done)
        self.worker.start()
        self.dialog.show()
        # self.dialog.cancel_button.clicked.connect(self.stop_thread)
        self.timer.start(1000)

    def coco_submit(self):
        data = {'dataset': 'coco'}
        data['str'] = self.coco_input.text()
        self.worker = SubmitThread(self, data)
        self.worker.finished.connect(self.done)
        self.worker.start()
        self.dialog.show()
        # self.dialog.cancel_button.clicked.connect(self.stop_thread)
        self.timer.start(1000)

    def done(self):
        self.dialog.close()
        self.timer.stop()

    def update_dialog(self):
        text = self.dialog.show_label.text()
        count = text.count('.')
        if count < 3:
            text += "."
        else:
            text = "Loading"
        self.dialog.show_label.setText(text)

    def stop_thread(self):
        self.worker.stop()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = Main()
    MainWindow.show()
    sys.exit(app.exec_())