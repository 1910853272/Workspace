# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/foucault/Projects/arc1_pyqt/uis/fitdialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_FitDialogParent(object):
    def setupUi(self, FitDialogParent):
        FitDialogParent.setObjectName("FitDialogParent")
        FitDialogParent.resize(730, 590)
        self.gridLayout = QtWidgets.QGridLayout(FitDialogParent)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(FitDialogParent)
        self.tabWidget.setObjectName("tabWidget")
        self.stimulusTab = QtWidgets.QWidget()
        self.stimulusTab.setObjectName("stimulusTab")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.stimulusTab)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.responsePlotWidget = PlotWidget(self.stimulusTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.responsePlotWidget.sizePolicy().hasHeightForWidth())
        self.responsePlotWidget.setSizePolicy(sizePolicy)
        self.responsePlotWidget.setMinimumSize(QtCore.QSize(400, 100))
        self.responsePlotWidget.setObjectName("responsePlotWidget")
        self.gridLayout_4.addWidget(self.responsePlotWidget, 0, 0, 1, 2)
        self.inputGroupBox = QtWidgets.QGroupBox(self.stimulusTab)
        self.inputGroupBox.setObjectName("inputGroupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.inputGroupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.refNegCombo = QtWidgets.QComboBox(self.inputGroupBox)
        self.refNegCombo.setObjectName("refNegCombo")
        self.gridLayout_2.addWidget(self.refNegCombo, 2, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.inputGroupBox)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 3, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.inputGroupBox)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 1, 0, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.inputGroupBox)
        self.lineEdit_9.setMouseTracking(False)
        self.lineEdit_9.setFocusPolicy(QtCore.Qt.NoFocus)
        self.lineEdit_9.setAcceptDrops(False)
        self.lineEdit_9.setStyleSheet("background-color: transparent;\n"
"color:transparent;\n"
"border: 2px solid transparent;")
        self.lineEdit_9.setText("a")
        self.lineEdit_9.setFrame(False)
        self.lineEdit_9.setReadOnly(True)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_2.addWidget(self.lineEdit_9, 5, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.inputGroupBox)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 2, 0, 1, 1)
        self.refPosCombo = QtWidgets.QComboBox(self.inputGroupBox)
        self.refPosCombo.setObjectName("refPosCombo")
        self.gridLayout_2.addWidget(self.refPosCombo, 1, 1, 1, 1)
        self.numPulsesEdit = QtWidgets.QLineEdit(self.inputGroupBox)
        self.numPulsesEdit.setObjectName("numPulsesEdit")
        self.gridLayout_2.addWidget(self.numPulsesEdit, 3, 1, 1, 1)
        self.parameterResultLabel = QtWidgets.QLabel(self.inputGroupBox)
        self.parameterResultLabel.setStyleSheet("color: red;\n"
"font-weight: bold;")
        self.parameterResultLabel.setText("")
        self.parameterResultLabel.setObjectName("parameterResultLabel")
        self.gridLayout_2.addWidget(self.parameterResultLabel, 0, 0, 1, 2)
        self.gridLayout_4.addWidget(self.inputGroupBox, 1, 0, 1, 1)
        self.resultsGroupBox = QtWidgets.QGroupBox(self.stimulusTab)
        self.resultsGroupBox.setObjectName("resultsGroupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.resultsGroupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.a0NegEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.a0NegEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.a0NegEdit.setReadOnly(True)
        self.a0NegEdit.setObjectName("a0NegEdit")
        self.gridLayout_3.addWidget(self.a0NegEdit, 3, 2, 1, 1)
        self.aNegEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.aNegEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.aNegEdit.setReadOnly(True)
        self.aNegEdit.setObjectName("aNegEdit")
        self.gridLayout_3.addWidget(self.aNegEdit, 1, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_8.setStyleSheet("font-weight: bold;")
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 0, 1, 1, 1)
        self.a0PosEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.a0PosEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.a0PosEdit.setReadOnly(True)
        self.a0PosEdit.setObjectName("a0PosEdit")
        self.gridLayout_3.addWidget(self.a0PosEdit, 3, 1, 1, 1)
        self.aPosEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.aPosEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.aPosEdit.setReadOnly(True)
        self.aPosEdit.setObjectName("aPosEdit")
        self.gridLayout_3.addWidget(self.aPosEdit, 1, 1, 1, 1)
        self.txNegEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.txNegEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.txNegEdit.setReadOnly(True)
        self.txNegEdit.setObjectName("txNegEdit")
        self.gridLayout_3.addWidget(self.txNegEdit, 2, 2, 1, 1)
        self.a1PosEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.a1PosEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.a1PosEdit.setReadOnly(True)
        self.a1PosEdit.setObjectName("a1PosEdit")
        self.gridLayout_3.addWidget(self.a1PosEdit, 4, 1, 1, 1)
        self.a1NegEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.a1NegEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.a1NegEdit.setReadOnly(True)
        self.a1NegEdit.setObjectName("a1NegEdit")
        self.gridLayout_3.addWidget(self.a1NegEdit, 4, 2, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 3, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 4, 0, 1, 1)
        self.txPosEdit = QtWidgets.QLineEdit(self.resultsGroupBox)
        self.txPosEdit.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.txPosEdit.setReadOnly(True)
        self.txPosEdit.setObjectName("txPosEdit")
        self.gridLayout_3.addWidget(self.txPosEdit, 2, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.resultsGroupBox)
        self.label_9.setStyleSheet("font-weight: bold;")
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 0, 2, 1, 1)
        self.gridLayout_4.addWidget(self.resultsGroupBox, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.fitButton = QtWidgets.QPushButton(self.stimulusTab)
        self.fitButton.setObjectName("fitButton")
        self.horizontalLayout.addWidget(self.fitButton)
        self.exportModelDataButton = QtWidgets.QPushButton(self.stimulusTab)
        self.exportModelDataButton.setObjectName("exportModelDataButton")
        self.horizontalLayout.addWidget(self.exportModelDataButton)
        self.exportVerilogButton = QtWidgets.QPushButton(self.stimulusTab)
        self.exportVerilogButton.setObjectName("exportVerilogButton")
        self.horizontalLayout.addWidget(self.exportVerilogButton)
        self.gridLayout_4.addLayout(self.horizontalLayout, 2, 0, 1, 2)
        self.tabWidget.addTab(self.stimulusTab, "")
        self.mechanismTab = QtWidgets.QWidget()
        self.mechanismTab.setObjectName("mechanismTab")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.mechanismTab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.mechanismTab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.modelStackedWidget = QtWidgets.QStackedWidget(self.groupBox_2)
        self.modelStackedWidget.setObjectName("modelStackedWidget")
        self.gridLayout_7.addWidget(self.modelStackedWidget, 1, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_2, 2, 2, 1, 1)
        self.mechanismPlotWidget = PlotWidget(self.mechanismTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mechanismPlotWidget.sizePolicy().hasHeightForWidth())
        self.mechanismPlotWidget.setSizePolicy(sizePolicy)
        self.mechanismPlotWidget.setObjectName("mechanismPlotWidget")
        self.gridLayout_5.addWidget(self.mechanismPlotWidget, 1, 0, 1, 3)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_5.addItem(spacerItem1, 0, 2, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.mechanismTab)
        self.label_11.setObjectName("label_11")
        self.gridLayout_5.addWidget(self.label_11, 0, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(self.mechanismTab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_6.addItem(spacerItem2, 1, 1, 1, 1)
        self.mechanismModelCombo = QtWidgets.QComboBox(self.groupBox)
        self.mechanismModelCombo.setObjectName("mechanismModelCombo")
        self.gridLayout_6.addWidget(self.mechanismModelCombo, 0, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setObjectName("label_12")
        self.gridLayout_6.addWidget(self.label_12, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox, 2, 0, 1, 2)
        self.curveSelectionSpinBox = QtWidgets.QSpinBox(self.mechanismTab)
        self.curveSelectionSpinBox.setMinimumSize(QtCore.QSize(50, 0))
        self.curveSelectionSpinBox.setObjectName("curveSelectionSpinBox")
        self.gridLayout_5.addWidget(self.curveSelectionSpinBox, 0, 1, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.fitMechanismModelButton = QtWidgets.QPushButton(self.mechanismTab)
        self.fitMechanismModelButton.setObjectName("fitMechanismModelButton")
        self.horizontalLayout_2.addWidget(self.fitMechanismModelButton)
        self.gridLayout_5.addLayout(self.horizontalLayout_2, 3, 0, 1, 3)
        self.gridLayout_5.setRowStretch(1, 3)
        self.gridLayout_5.setRowStretch(2, 2)
        self.tabWidget.addTab(self.mechanismTab, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.retranslateUi(FitDialogParent)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(FitDialogParent)
        FitDialogParent.setTabOrder(self.fitButton, self.exportModelDataButton)

    def retranslateUi(self, FitDialogParent):
        _translate = QtCore.QCoreApplication.translate
        FitDialogParent.setWindowTitle(_translate("FitDialogParent", "Dialog"))
        self.inputGroupBox.setTitle(_translate("FitDialogParent", "Parameter Fit"))
        self.label_3.setText(_translate("FitDialogParent", "Pulses per set"))
        self.label.setText(_translate("FitDialogParent", "Reference Positive Voltage (V)"))
        self.label_2.setText(_translate("FitDialogParent", "Refence Negative Voltage (V)"))
        self.numPulsesEdit.setText(_translate("FitDialogParent", "500"))
        self.resultsGroupBox.setTitle(_translate("FitDialogParent", "Results"))
        self.label_8.setText(_translate("FitDialogParent", "Positive voltages"))
        self.label_4.setText(_translate("FitDialogParent", "A"))
        self.label_5.setText(_translate("FitDialogParent", "tx"))
        self.label_6.setText(_translate("FitDialogParent", "a0"))
        self.label_7.setText(_translate("FitDialogParent", "a1"))
        self.label_9.setText(_translate("FitDialogParent", "Negative Voltages"))
        self.fitButton.setText(_translate("FitDialogParent", "Fit Stimulus Model"))
        self.exportModelDataButton.setText(_translate("FitDialogParent", "Export Stimulus Model data"))
        self.exportVerilogButton.setText(_translate("FitDialogParent", "Export Verilog-A Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.stimulusTab), _translate("FitDialogParent", "Stimulus"))
        self.groupBox_2.setTitle(_translate("FitDialogParent", "Results"))
        self.label_11.setText(_translate("FitDialogParent", "Curve selection:"))
        self.groupBox.setTitle(_translate("FitDialogParent", "Parameter fit for selected curve"))
        self.label_12.setText(_translate("FitDialogParent", "Model"))
        self.fitMechanismModelButton.setText(_translate("FitDialogParent", "Fit Mechanism Model"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.mechanismTab), _translate("FitDialogParent", "Mechanism"))
from pyqtgraph import PlotWidget
