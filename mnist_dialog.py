# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mnist.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(400, 300)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 250, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.btn1 = QtWidgets.QPushButton(Dialog)
        self.btn1.setGeometry(QtCore.QRect(30, 10, 331, 29))
        self.btn1.setObjectName("btn1")
        self.btn2 = QtWidgets.QPushButton(Dialog)
        self.btn2.setGeometry(QtCore.QRect(30, 50, 331, 29))
        self.btn2.setObjectName("btn2")
        self.btn3 = QtWidgets.QPushButton(Dialog)
        self.btn3.setGeometry(QtCore.QRect(30, 90, 331, 29))
        self.btn3.setObjectName("btn3")
        self.btn4 = QtWidgets.QPushButton(Dialog)
        self.btn4.setGeometry(QtCore.QRect(30, 130, 331, 29))
        self.btn4.setObjectName("btn4")
        self.btn5 = QtWidgets.QPushButton(Dialog)
        self.btn5.setGeometry(QtCore.QRect(30, 210, 331, 29))
        self.btn5.setObjectName("btn5")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(30, 180, 141, 19))
        self.label.setObjectName("label")
        self.lineEdit = QtWidgets.QLineEdit(Dialog)
        self.lineEdit.setGeometry(QtCore.QRect(170, 170, 181, 29))
        self.lineEdit.setObjectName("lineEdit")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.btn1.setText(_translate("Dialog", "5.1 Show Train Images"))
        self.btn2.setText(_translate("Dialog", "5.2 Show Hyperparameters"))
        self.btn3.setText(_translate("Dialog", "5.3 Train Epoch"))
        self.btn4.setText(_translate("Dialog", "5.4 Show Training Result"))
        self.btn5.setText(_translate("Dialog", "5.5 Inference"))
        self.label.setText(_translate("Dialog", "Test Image Index:"))
