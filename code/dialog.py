# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/Users/samuel/DL_hw/dialog.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(380, 115)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.space = QtWidgets.QLabel(Dialog)
        self.space.setText("")
        self.space.setObjectName("space")
        self.gridLayout.addWidget(self.space, 0, 1, 1, 1)
        self.show_label = QtWidgets.QLabel(Dialog)
        self.show_label.setAlignment(QtCore.Qt.AlignCenter)
        self.show_label.setObjectName("show_label")
        self.gridLayout.addWidget(self.show_label, 0, 0, 1, 2)
        self.cancel_button = QtWidgets.QPushButton(Dialog)
        self.cancel_button.setObjectName("cancel_button")
        self.gridLayout.addWidget(self.cancel_button, 1, 1, 1, 1)
        self.gridLayout.setColumnStretch(0, 10)
        self.gridLayout.setColumnStretch(1, 4)
        self.gridLayout.setColumnStretch(2, 1)
        self.gridLayout.setRowStretch(0, 1)
        self.gridLayout.setRowStretch(1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Loading"))
        self.show_label.setText(_translate("Dialog", "Loading"))
        self.cancel_button.setText(_translate("Dialog", "Cancel"))

