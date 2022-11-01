
import os
import sys
import cv2
import os.path
import numpy as np
import mediapipe as mp
import uuid
from PyQt5.QtCore import pyqtSlot, QPoint, QDate
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from PyQt5.QtWidgets import QApplication, QListWidget, QGroupBox, QCheckBox, QComboBox, QLabel, QLineEdit, QDialog, \
    QDialogButtonBox, QTabWidget, QWidget, QVBoxLayout, QGridLayout
import sys
from PyQt5.QtGui import QIcon, QFont

from os import path
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QInputDialog, QMainWindow, \
    QDateEdit, QSpinBox, QComboBox, QCheckBox, QLineEdit
from PyQt5.QtGui import QPixmap, QRegExpValidator
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QRegExp
from datetime import datetime
from random import randint

from Angle_Calculater_finger import Angle_Calculator_finger
from Angle_Calc_Right_ELBOW import Angle_Calc_Right_ELBOW
from Angle_Calc_Gunshot import Angle_Calc_Gunshot
from Angle_Calc_Left_ELBOW import Angle_Calc_Left_ELBOW

from Angle_Calc_Left_KNEE import Angle_Calc_Left_KNEE
from Angle_Calc_Right_KNEE import Angle_Calc_Right_KNEE

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

SITE_OF_FRACTURE = ["TBD","HAND:Finger(ThumbTip-Wrist-IndexFingerTip)","HAND:Finger(IndexFingerTip-MiddleFingerMCP-MiddleFingerTip)","HAND:Finger(MiddleFingerTip-RingFingerMCP-RingFingerTip)",
                    "HAND:Finger(RingFingerTip-PinkyMCP-PinkyTip)", "ARM:Right-Elbow", "ARM:LowerArm(GunShot)", "ARM:Left-Elbow" , "LEG:Left-Knee" , "LEG:Right-Knee"]
#TYPE_OF_FRACTURE = ["TBD","FINGER", "RIGHT HAND", "LEFT HAND", "GUN SHOT"]
TYPE_OF_FRACTURE = ["TBD","Stable/Closed", "Open/Compund", "Transverse", "Oblique", "Comminuted"]
COMPLICATION = ["TBD","YES", "NO"]
#TYPE_OF_COMPLICATION = ["TBD","Stable Non Dislocated", "Open / Compound Fracture", "Transverse", "Oblique", "Comminuted", "Odema", "Vascular Injury", "Nerve Damage", "Local infection", "Compartment Syndorm" ]
TYPE_OF_COMPLICATION = ["TBD","Odema", "Vascular Injury", "Nerve Damage", "Local Infection", "Osteomyelitis", "Delayed Union", "Malunion", "Non-Union", "Local Pain Syndrom", "Avascular Necrosis", "Local Reaction to Fixation Device" ]
COMPLICARION_FOLLOWUP = ["TBD","Infection", "Osteomyelitis", "Delayed Union", "Malunion", "Non Union", "Local Pain Syndorm", "Avascular Necrosis", "Local Reaction to Fixation Device", "Avascular Necrosis", "Joint Stiffness", "Sudeck's Dystrophy", "Ischemic Osteomyelitis"]
OUTPUT_DIR = 'Output Images'
evaluatedValue = 0

supinationPronation = 0
isEvaSet_G = False

hiddenimports = [
    "VideoProcessingThread.py",
    "Angle_Calculater_finger.py",
                 "Angle_Calc_Right_ELBOW.py",
                 "Angle_Calc_Left_ELBOW.py",
                 "Angle_Calc_Left_KNEE",
                "Angle_Calc_Right_KNEE",
                 "Angle_Calc_Gunshot.py",
                 "ImageConverter.py",
                 "local_constants.py",
                 "qtoggle.py",
                 "RoM_Reports.py",
                 "utils.py"
                 ]

added_files = [
         ( 'C:\Program Files\Python_3.8\Lib\site-packages\mediapipe\modules',
           'mediapipe\modules' ),
         ( '*.md', '.' )
         ]

class VideoProcessingThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_elbow_eval_values_signal = pyqtSignal(object)
    change_finger_eval_values_signal = pyqtSignal(object)
    change_gunshot_eval_values_signal = pyqtSignal(object)
    change_setDATA = pyqtSignal(object)
    change_setDATA_2 = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._run_flag = True

        self.imageutil = ImageConverter()

        self.mp_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.angle_calculator_finger = Angle_Calculator_finger()
        self.angle_Calc_Right_ELBOW = Angle_Calc_Right_ELBOW()
        self.angle_Calc_Gunshot = Angle_Calc_Gunshot()
        self.angle_Calc_Left_ELBOW = Angle_Calc_Left_ELBOW()
        self.Angle_Calc_Left_KNEE = Angle_Calc_Left_KNEE()
        self.Angle_Calc_Right_KNEE = Angle_Calc_Right_KNEE()

        self.body_part_selection = ""
        self.isStartSet=False
        self.evaluation = True
        self.Supination_Pronation = True
        self.eval_values_dict = {}

    def run(self):
        # capture from web cam
        try:
            print("Video thread started")
            cap = cv2.VideoCapture(0)
            while self._run_flag:
                ret, cv_img = cap.read()
                #print("Size of the image")
                #print(cv_img.shape)
                #scale_percent = 500  # percent of original size
                # width = int(cv_img.shape[1] * scale_percent / 100)
                # height = int(cv_img.shape[0] * scale_percent / 100)
                # width = int(cv_img.shape[1] * 1)
                # height = int(cv_img.shape[0] * 1)
                # dim = (width, height)
                # # resize image
                # resized = cv2.resize(cv_img, dim, interpolation=cv2.INTER_AREA)
                # cv_img = resized
                #print('Resized Dimensions (source image) : ', resized.shape)
                if ret:
                    # evaluate pose/hand
                    self.processed_cv_image = self.evaluate_image_with_mp(cv_img)
                    self.change_pixmap_signal.emit(self.processed_cv_image)

            # shut down capture system
            cap.release()
        except:
            print("An Error occured in Video Processing Thread")

    def evaluate_image_with_mp(self, cv_img):
        global evaluatedValue
        #print("sambath - datachange")
        #print(self.evaluation)
        #print("sambath - datachange")
        try:
            if self.body_part_selection == "HAND:Finger(ThumbTip-Wrist-IndexFingerTip)":
                self.AngSel_BWFinger = 1
                processed_image = self.angle_calculator_finger.process(cv_img ,self.AngSel_BWFinger )
                self.change_finger_eval_values_signal.emit(self.angle_calculator_finger.get_eval_values())
                return processed_image

            if self.body_part_selection == "HAND:Finger(IndexFingerTip-MiddleFingerMCP-MiddleFingerTip)":
                self.AngSel_BWFinger = 2
                processed_image = self.angle_calculator_finger.process(cv_img ,self.AngSel_BWFinger )
                self.change_finger_eval_values_signal.emit(self.angle_calculator_finger.get_eval_values())
                return processed_image

            if self.body_part_selection == "HAND:Finger(MiddleFingerTip-RingFingerMCP-RingFingerTip)":
                self.AngSel_BWFinger = 3
                processed_image = self.angle_calculator_finger.process(cv_img ,self.AngSel_BWFinger )
                self.change_finger_eval_values_signal.emit(self.angle_calculator_finger.get_eval_values())
                return processed_image

            if self.body_part_selection == "HAND:Finger(RingFingerTip-PinkyMCP-PinkyTip)":
                self.AngSel_BWFinger = 4
                processed_image = self.angle_calculator_finger.process(cv_img ,self.AngSel_BWFinger )
                self.change_finger_eval_values_signal.emit(self.angle_calculator_finger.get_eval_values())
                return processed_image

            if self.body_part_selection == "ARM:Right-Elbow":
                rightHand = self.angle_Calc_Right_ELBOW.process(cv_img, self.isStartSet, self.evaluation)
                self.change_elbow_eval_values_signal.emit(self.angle_Calc_Right_ELBOW.get_eval_values())
                return rightHand

            if self.body_part_selection == "ARM:LowerArm(GunShot)":
                gunShot =  self.angle_Calc_Gunshot.process(cv_img, self.isStartSet, self.evaluation )
                self.change_gunshot_eval_values_signal.emit(self.angle_Calc_Gunshot.get_eval_values())
                return gunShot

            if self.body_part_selection == "ARM:Left-Elbow":
                leftHand = self.angle_Calc_Left_ELBOW.process(cv_img, self.isStartSet, self.evaluation)
                self.change_elbow_eval_values_signal.emit(self.angle_Calc_Left_ELBOW.get_eval_values())
                return leftHand
            if self.body_part_selection == "LEG:Left-Knee":
                leftHand = self.Angle_Calc_Left_KNEE.process(cv_img, self.isStartSet, self.evaluation)
                self.change_elbow_eval_values_signal.emit(self.Angle_Calc_Left_KNEE.get_eval_values())
                return leftHand
            if self.body_part_selection == "LEG:Right-Knee":
                leftHand = self.Angle_Calc_Right_KNEE.process(cv_img, self.isStartSet, self.evaluation)
                self.change_elbow_eval_values_signal.emit(self.Angle_Calc_Right_KNEE.get_eval_values())
                return leftHand
            return self.imageutil.flip(cv_img)
        except:
            print("An Error occured ")

    def on_update_body_part_selection(self, text):
        #print("on_update_body_part_selection")
        if text in SITE_OF_FRACTURE:
            self.body_part_selection = text

    def on_update_start_stop(self, isStartSet):
        self.isStartSet = isStartSet

    def on_Supination_Pronation(self, isStartSet):
        global supinationPronation
        self.isSupination_Pronation = isStartSet
        supinationPronation = isStartSet

    def on_update_Evaluate(self, isEvaSet):
        self.evaluation = isEvaSet

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class ImageConverter():

    def convert_cv_2_rgb(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        return rgb_image

    def flip(self, cv_img):
        # BGR 2 RGB
        self.image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        # Flip on horizontal
        self.image = cv2.flip(self.image, 1)
        # RGB 2 BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        return self.image

    def convert_cv_2_rgb_qt(self, cv_img, disply_width = 900, display_height = 1400):
        """Convert from an opencv image to QPixmap"""
        rgb_image = self.convert_cv_2_rgb(cv_img)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(disply_width, display_height, Qt.KeepAspectRatio)
        return rgb_image, QPixmap.fromImage(p)

class App (QWidget):

    def __init__(self):
        super().__init__()
        self.disply_width = 1000
        self.display_height = 480
        self.init_gui()
        #self.is_store_images = False

    def init_gui(self):
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('Webcam')
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        tabwidget = QTabWidget()
        tabwidget.setFont((QFont('Arial', 13,QFont.Bold)))
        tabwidget.setStyleSheet("color: black;  background-color: white")
        tabwidget.colorCount()
        self.thread = VideoProcessingThread()
        self.firstTab =FirstTab(self.thread)
        self.secondTab = SecondTab(self.thread)
        self.thirdTab = ThirdTab(self.thread)
        self.fourthTab =FourthTab(self.thread)
        self.fifthTab =FifthTab(self.thread)
        self.sixthTab = SixthTab()

        tabwidget.addTab(self.firstTab," Patient Information ",)
        #tabwidget.addTab(self.secondTab," Tretment Information ")
        tabwidget.addTab(self.secondTab, " Clinical Information ")
        tabwidget.addTab(self.thirdTab ," Camera Capture ")
        tabwidget.addTab(self.fourthTab, " ROM1 ")
        #tabwidget.addTab(self.fifthTab, " Supination-Pronation Measurement Result ")
        #tabwidget.addTab(self.sixthTab, " Treatment Recommendation ")
        tabwidget.addTab(self.fifthTab, " ROM2 ")
        tabwidget.addTab(self.sixthTab, " Recommendation ")

        vbox.addWidget(tabwidget)
        vbox.addWidget(self.image_label)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

class FirstTab (QWidget):
    def __init__(self, video_thread):
        super().__init__()
        self.Text_FirstTab = {}

        self.Patient_ID = QLabel("Patient ID: ")
        self.Patient_ID.setFont(QFont('Arial', 15))
        self.PatientID_Edit = QLineEdit()
        self.PatientID_Edit.setFont(QFont('Arial', 12))
        #id = uuid.uuid4()
        self.PatientID_Edit.setText(str(uuid.uuid4()))
        #self.baba = self.PatientID_Edit.text()
        self.DOR = QLabel("Date Of Registration: ")
        self.DOR.setFont(QFont('Arial', 15))

        self.Cala = QPushButton("Calender")
        #self.Cala.setFont(QFont('Arial', 15))
        self.Cala.setStyleSheet("background-color: lightgray")
        self.Cala.clicked.connect(self.popUpcal)

        self.DOR_Edit = QLineEdit()
        self.DOR_Edit.setFont(QFont('Arial', 12))
        #date = self.calendar.selectedDate()
        #print(date)
        self.DOR_Edit.setText(str("TBD "))
        #self.DOR_Edit.setStyleSheet("color: red;")

        self.F_Name = QLabel("First Name: ")
        self.F_Name.setFont(QFont('Arial', 15))
        self.F_Name_Edit = QLineEdit()
        self.F_Name_Edit.setFont(QFont('Arial', 12))
        self.F_Name_Edit.setText(str("TBD "))
        self.L_Name = QLabel("Last Name: ")
        self.L_Name.setFont(QFont('Arial', 15))
        self.L_Name_Edit = QLineEdit()
        self.L_Name_Edit.setFont(QFont('Arial', 12))
        self.L_Name_Edit.setText(str(" TBD"))
        self.DOB   = QLabel("Date of Birth: ")
        self.DOB.setFont(QFont('Arial', 15))
        self.DOB_Edit = QLineEdit()
        self.DOB_Edit.setFont(QFont('Arial', 12))

        #self.calendar = Calendar(self)
        self.Cala_DOB = QPushButton("Calender")
        #self.Cala.setFont(QFont('Arial', 15))
        self.Cala_DOB.setStyleSheet("background-color: lightgray")
        self.Cala_DOB.clicked.connect(self.popUpcal_Dob)

        self.DOB_Edit.setText(str("TBD "))
        self.Gender   = QLabel("Gender: ")
        self.Gender.setFont(QFont('Arial', 15))
        self.Gender_Edit = QLineEdit()
        self.Gender_Edit.setFont(QFont('Arial', 12))
        self.Gender_Edit.setText(str(" TBD"))
        self.Address   = QLabel("Address: ")
        self.Address.setFont(QFont('Arial', 15))
        self.Address_Edit = QLineEdit()
        self.Address_Edit.setFont(QFont('Arial', 12))
        self.Address_Edit.setText(str("TBD "))
        self.Phone   = QLabel("Phone: ")
        self.Phone.setFont(QFont('Arial', 15))
        self.Phone_Edit = QLineEdit()
        self.Phone_Edit.setFont(QFont('Arial', 12))
        self.Phone_Edit.setText(str("TBD "))
        self.E_Mail   = QLabel("E-Mail: ")
        self.E_Mail.setFont(QFont('Arial', 15))
        self.E_Mail_Edit = QLineEdit()
        self.E_Mail_Edit.setText(str("TBD "))
        self.E_Mail_Edit.setFont(QFont('Arial', 12))
        layout = QGridLayout()
        layout.addWidget(self.Patient_ID)

        layout.addWidget(self.PatientID_Edit)
        #layout.addWidget(self.Cala)
        layout.addWidget(self.DOR)
        layout.addWidget(self.Cala)
        layout.addWidget(self.DOR_Edit)
        layout.addWidget(self.F_Name)
        layout.addWidget(self.F_Name_Edit)
        layout.addWidget(self.L_Name)
        layout.addWidget(self.L_Name_Edit)
        layout.addWidget(self.DOB)
        layout.addWidget(self.Cala_DOB)
        layout.addWidget(self.DOB_Edit)
        layout.addWidget(self.Gender)
        layout.addWidget(self.Gender_Edit)
        layout.addWidget(self.Address)
        layout.addWidget(self.Address_Edit)
        layout.addWidget(self.Phone)
        layout.addWidget(self.Phone_Edit)
        layout.addWidget(self.E_Mail)
        layout.addWidget(self.E_Mail_Edit)

        self.save = QPushButton("SAVE Patient Information")
        self.save.setStyleSheet("background-color: lightgreen")

        layout.addWidget(self.save)
        self.setLayout(layout)
        self.thread = video_thread
        self.saveInput_PI()
        self.save.clicked.connect(self.saveInput_PI)

    def saveInput_PI(self):
        print("save - emit")
        self.thread.change_setDATA.emit(self.set_FirstTabValue())

    def popUpcal(self):
            #print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate)

    def showDate(self, date):
        print(date.toString())
        self.DOR_Edit.setText(date.toString())

    def popUpcal_Dob(self):
            print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate_Dob)

    def showDate_Dob(self, date):
        print(date.toString())
        self.DOB_Edit.setText(date.toString())

    def set_FirstTabValue(self):
        print("save - data")
        self.Text_FirstTab["Patient_ID"]     =self.Patient_ID.text()
        self.Text_FirstTab["PatientID_Edit"] =self.PatientID_Edit.text()
        self.Text_FirstTab["DOR"]            =self.DOR.text()
        self.Text_FirstTab["DOR_Edit"]       =self.DOR_Edit.text()
        self.Text_FirstTab["F_Name"]         =self.F_Name.text()
        self.Text_FirstTab["F_Name_Edit"]    =self.F_Name_Edit.text()
        self.Text_FirstTab["L_Name"]         =self.L_Name.text()
        self.Text_FirstTab["L_Name_Edit"]    =self.L_Name_Edit.text()
        self.Text_FirstTab["DOB"]            =self.DOB.text()
        self.Text_FirstTab["DOB_Edit"]       =self.DOB_Edit.text()
        self.Text_FirstTab["Gender"]         =self.Gender.text()
        self.Text_FirstTab["Gender_Edit"]    =self.Gender_Edit.text()
        self.Text_FirstTab["Address"]        =self.Address.text()
        self.Text_FirstTab["Address_Edit"]   =self.Address_Edit.text()
        self.Text_FirstTab["Phone"]          =self.Phone.text()
        self.Text_FirstTab["Phone_Edit"]     =self.Phone_Edit.text()
        self.Text_FirstTab["E_Mail"]         =self.E_Mail.text()
        self.Text_FirstTab["E_Mail_Edit"]    =self.E_Mail_Edit.text()
        print("data RETURN CALLED")
        return self.Text_FirstTab

class SecondTab (QWidget):
    def __init__(self, video_thread):
        super().__init__()
        self.Text_SecondTab = {}
        layout = QVBoxLayout()
        self.SOF_Edit_Selected_1 = "TTD"
        self.TOF_Edit_Selected_1 = "TTD"
        self.Complication_Edit_Selected_1 = "TTD"
        self.TOC_Edit_Selected_1 = "TTD"
        self.CFU_Edit_Selected_1 = "TTD"

        self.DOF = QLabel("Date of Fracture: ")
        self.DOF.setFont(QFont('Arial', 15))
        self.DOF_Edit = QLineEdit()
        self.Cala_DOF = QPushButton("Calender")
        #self.Cala.setFont(QFont('Arial', 15))
        self.Cala_DOF.setStyleSheet("background-color: lightgray")
        self.Cala_DOF.clicked.connect(self.popUpcal_Dof)
        self.DOF_Edit.setFont(QFont('Arial', 12))
        self.DOF_Edit.setText(str("TBD"))

        layout.addWidget(self.DOF)
        layout.addWidget(self.Cala_DOF)
        layout.addWidget(self.DOF_Edit)

        self.SOF   = QLabel("Site of Fracture: ")
        self.SOF.setFont(QFont('Arial', 15))
        self.SOF_Edit = QComboBox()
        self.SOF_Edit.setGeometry(QtCore.QRect(0, 0, 130, 60))
        self.SOF_Edit.setFont(QFont('Arial', 12))
        self.SOF_Edit.addItems(SITE_OF_FRACTURE)
        self.SOF_Edit.setCurrentIndex(0)
        #self.SOF_Edit.setCurrentText(str(self.SOF_Edit.setCurrentIndex(0)))
        #self.SOF_Edit.currentText()
        #self.SOF_Edit.activated[str].connect(self.Selected_SOF_Edit)
        self.SOF_Edit.activated[str].connect(self.showSelectedBodyPart)
        layout.addWidget(self.SOF)
        layout.addWidget(self.SOF_Edit)

        self.TOF   = QLabel("Type of Fracture: ")
        self.TOF.setFont(QFont('Arial', 15))
        #TOF_Edit = QLineEdit()
        self.TOF_Edit = QComboBox()
        self.TOF_Edit.setGeometry(QtCore.QRect(0, 0, 130, 60))
        self.TOF_Edit.setFont(QFont('Arial', 12))
        self.TOF_Edit.addItems(TYPE_OF_FRACTURE)
        self.TOF_Edit.setCurrentIndex(0)
        #self.TOF_Edit.setCurrentText(str(self.TOF_Edit.setCurrentIndex(0)))
        #self.TOF_Edit.setCurrentText(self, "TBD")
        self.TOF_Edit.activated[str].connect(self.Selected_TOF_Edit)

        layout.addWidget(self.TOF)
        layout.addWidget(self.TOF_Edit)

        self.Complication   = QLabel("Complication(YES/NO): ")
        self.Complication.setFont(QFont('Arial', 15))
        self.Complication_Edit = QComboBox()
        self.Complication_Edit.setGeometry(QtCore.QRect(0, 0, 130, 60))
        self.Complication_Edit.setFont(QFont('Arial', 12))
        self.Complication_Edit.addItems(COMPLICATION)
        self.Complication_Edit.setCurrentIndex(0)
        #self.Complication_Edit.setCurrentText(str(self.Complication_Edit.setCurrentIndex(0)))
        #self.Complication_Edit.setCurrentText()
        self.Complication_Edit.activated[str].connect(self.Selected_Complication_Edit)
        #Complication_Edit = QLineEdit()

        layout.addWidget(self.Complication)
        layout.addWidget(self.Complication_Edit)

        self.TOC   = QLabel("Type of Complication: ")
        self.TOC.setFont(QFont('Arial', 15))
        #TOC_Edit = QLineEdit()
        self.TOC_Edit = QComboBox()
        self.TOC_Edit.setGeometry(QtCore.QRect(0, 0, 130, 60))
        self.TOC_Edit.setFont(QFont('Arial', 12))
        self.TOC_Edit.addItems(TYPE_OF_COMPLICATION)
        self.TOC_Edit.setCurrentIndex(0)
        #self.TOC_Edit.setCurrentText(str(self.TOC_Edit.setCurrentIndex(0)))
        #self.TOC_Edit.setCurrentText()
        self.TOC_Edit.activated[str].connect(self.Selected_TOC_Edit)

        layout.addWidget(self.TOC)
        layout.addWidget(self.TOC_Edit)

        self.DOT = QLabel("Date Of Treatment: ")
        self.DOT.setFont(QFont('Arial', 15))
        self.DOT_Edit = QLineEdit()
        self.Cala_DOT = QPushButton("Calender")
        self.Cala_DOT.setStyleSheet("background-color: lightgray")
        self.Cala_DOT.clicked.connect(self.popUpcal_Dot)
        self.DOT_Edit.setFont(QFont('Arial', 12))
        self.DOT_Edit.setText(str("TBD"))

        layout.addWidget(self.DOT)
        layout.addWidget(self.Cala_DOT)
        layout.addWidget(self.DOT_Edit)

        self.TOT = QLabel("Type Of Treatment: ")
        self.TOT.setFont(QFont('Arial', 15))
        self.TOT_Edit = QLineEdit()
        self.TOT_Edit.setFont(QFont('Arial', 12))
        self.TOT_Edit.setText(str("TBD"))

        layout.addWidget(self.TOT)
        layout.addWidget(self.TOT_Edit)

        self.DOTFU = QLabel("Date of Treatment Follow up: ")
        self.DOTFU.setFont(QFont('Arial', 15))
        self.DOTFU_Edit = QLineEdit()
        self.DOTFU_Edit.setFont(QFont('Arial', 12))
        self.DOTFU_Edit.setText(str("TBD"))
        self.Cala_DTF = QPushButton("Calender")
        self.Cala_DTF.setStyleSheet("background-color: lightgray")
        self.Cala_DTF.clicked.connect(self.popUpcal_CFU)

        layout.addWidget(self.DOTFU)
        layout.addWidget(self.Cala_DTF)
        layout.addWidget(self.DOTFU_Edit)

        self.CFU   = QLabel("Complication Follow-Up: ")
        self.CFU.setFont(QFont('Arial', 15))
        #CFU_Edit = QLineEdit()

        # self.CFU_Edit = QComboBox()
        # self.CFU_Edit.setGeometry(QtCore.QRect(0, 0, 130, 60))
        # self.CFU_Edit.setFont(QFont('Arial', 12))
        # self.CFU_Edit.addItems(COMPLICARION_FOLLOWUP)
        # self.CFU_Edit.setCurrentIndex(0)
        # #self.CFU_Edit.setCurrentText((str("ttr")))
        # #self.CFU_Edit.setCurrentText()
        # #self.CFU_Edit.activated[str].connect(self.showSelectedBodyPart)
        # self.CFU_Edit.activated[str].connect(self.Selected_CFU_Edit)
        self.Cala_CFU = QPushButton("Calender")
        self.Cala_CFU.setStyleSheet("background-color: lightgray")
        self.Cala_CFU.clicked.connect(self.popUpcal_DCFU)
        self.CFU_Edit = QLineEdit()
        self.CFU_Edit.setFont(QFont('Arial', 12))
        self.CFU_Edit.setText(str("TBD"))
        layout.addWidget(self.CFU)
        layout.addWidget(self.Cala_CFU)
        layout.addWidget(self.CFU_Edit)


        self.save = QPushButton("SAVE Treatment Information")
        self.save.setStyleSheet("background-color: lightgreen")

        layout.addWidget(self.save)
        #self.setLayout(layout)
        self.thread = video_thread
        self.saveInput_PI()
        self.save.clicked.connect(self.saveInput_PI)
        self.setLayout(layout)

    def showSelectedBodyPart(self, text):
        #print("Selected Body Part is : " + text)
        self.thread.on_update_body_part_selection(text)

    def Selected_SOF_Edit(self, text):
        #print("sambath : " + text)
        self.SOF_Edit_Selected_1 = text

    def Selected_TOF_Edit(self, text):
        #print("sambath : " + text)
        self.TOF_Edit_Selected_1 = text

    def Selected_Complication_Edit(self, text):
        #print("sambath : " + text)
        self.Complication_Edit_Selected_1 = text

    def Selected_TOC_Edit(self, text):
        #print("sambath : " + text)
        self.TOC_Edit_Selected_1 = text

    def Selected_CFU_Edit(self, text):
        #print("sambath : " + text)
        self.CFU_Edit_Selected_1 = text

    def saveInput_PI(self):
        print("second Tab save - emit")
        self.thread.change_setDATA_2.emit(self.set_SecondTabValue())
        #print("when itis called")

    def popUpcal_Dof(self):
            print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate_Dof)

    def showDate_Dof(self, date):
        #print(date.toString())
        self.DOF_Edit.setText(date.toString())

    def popUpcal_Dot(self):
            #print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate_Dot)

    def showDate_Dot(self, date):
        #print(date.toString())
        self.DOT_Edit.setText(date.toString())

    def popUpcal_CFU(self):
            print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate_CFU)

    def showDate_CFU(self, date):
        print(date.toString())
        self.DOTFU_Edit.setText(date.toString())

    def popUpcal_DCFU(self):
            #print("no external windows")
            self.w = AnotherWindow()
            self.w.show()
            self.calendar = self.w.getCalender()
            self.calendar.clicked[QDate].connect(self.showDate_DCFU)

    def showDate_DCFU(self, date):
        #print(date.toString())
        self.CFU_Edit.setText(date.toString())

    def set_SecondTabValue(self):
        print("second data published")
        self.Text_SecondTab["DOF"] = self.DOF.text()
        self.Text_SecondTab["DOF_Edit"] = self.DOF_Edit.text()
        self.Text_SecondTab["SOF"] = self.SOF.text()
        self.Text_SecondTab["SOF_Edit"] = self.SOF_Edit_Selected_1
        self.Text_SecondTab["TOF"] = self.TOF.text()
        self.Text_SecondTab["TOF_Edit"] = self.TOF_Edit_Selected_1
        self.Text_SecondTab["Complication"] = self.Complication.text()
        self.Text_SecondTab["Complication_Edit"] = self.Complication_Edit_Selected_1
        self.Text_SecondTab["TOC"] = self.TOC.text()
        self.Text_SecondTab["TOC_Edit"] = self.TOC_Edit_Selected_1
        self.Text_SecondTab["DOT"] = self.DOT.text()
        self.Text_SecondTab["DOT_Edit"] = self.DOT_Edit.text()
        self.Text_SecondTab["TOT"] = self.TOT.text()
        self.Text_SecondTab["TOT_Edit"] = self.TOT_Edit.text()
        self.Text_SecondTab["DOTFU"] = self.DOTFU.text()
        self.Text_SecondTab["DOTFU_Edit"] = self.DOTFU_Edit.text()
        self.Text_SecondTab["CFU"] = self.CFU.text()
        self.Text_SecondTab["CFU_Edit"] = self.CFU_Edit_Selected_1
        return self.Text_SecondTab

class ThirdTab (QWidget):
    def __init__(self, video_thread):
        super().__init__()
               # self._date_widget = QDateEdit()
        # self._date_widget.setCalendarPopup(True)
        # FRACTURE DATE
        # create the label that holds the image
        #self.disply_width = 1920
        #self.display_height = 1080
        #self.disply_width = 20
        #self.display_height = 20
        self.image_label = QLabel()
        #self.image_label.move(self, "Center")
        #self.image_label.resize(self.disply_width, self.display_height)



        vbox = QVBoxLayout()
        Hbox = QHBoxLayout()

        self.TOT_Edit = QLineEdit()
        self.TOT_Edit.setFont(QFont('Arial', 12))
        self.TOT_Edit.setText(str("Recommended Distance between Patient and Camera based on the Patient's height - TBD"))
        vbox.addWidget(self.TOT_Edit)

        self.checkbox_evaluation = QCheckBox("Evaluate_Angle")
        self.checkbox_evaluation.setFont(QFont('Arial', 10))
        #self.evaluation(self.checkbox_evaluation)
        self.checkbox_evaluation.stateChanged.connect(lambda: self.evaluation(self.checkbox_evaluation))

        # self.fractureSince_textLabel = QLabel('Number of days since Fractured')
        # self.input_box = QSpinBox()
        # self.input_box.resize(200, 20)
        # SELECT FRACTURED BODY PART
        # self.lbl_select_body_part = QLabel("Which part of the body is Fractured?")
        # self.lbl_select_body_part.setFont(QFont('Arial', 10))
        # self.dd_select_body_part = QComboBox()
        # self.dd_select_body_part.setGeometry(QtCore.QRect(0, 0, 130, 60))
        # self.dd_select_body_part.setFont(QFont('Arial', 10))
        # self.dd_select_body_part.addItems(BODY_PARTS)
        # self.dd_select_body_part.setCurrentIndex(0)
        # self.dd_select_body_part.activated[str].connect(self.showSelectedBodyPart)

        # Character Input A-Z Input
        #self.lbl_name = QLabel("ENTER THE NAME")

        # self.input_name = QLineEdit(self)
        # self.reg = QRegExp("[A-Za-z_]+")
        # self.validar_str = QRegExpValidator(self.reg)
        # self.input_name.setValidator(self.validar_str)

        # check box - Start/Stop
        self.checkbox_startstop = QCheckBox("Start")
        self.checkbox_startstop.setFont(QFont('Arial', 10))
        self.checkbox_startstop.stateChanged.connect(lambda: self.update_start_stop(self.checkbox_startstop))

        self.checkbox_Supination_Pronation = QCheckBox("Supination")
        self.checkbox_Supination_Pronation.setFont(QFont('Arial', 10))
        self.checkbox_Supination_Pronation.stateChanged.connect(lambda: self.update_Supination_Pronation(self.checkbox_Supination_Pronation))

        # check box - (De)/Activate report generator
        self.checkbox_ActivateReportGenerator = QCheckBox("Activate report generator")
        self.checkbox_ActivateReportGenerator.stateChanged.connect(
            lambda: self.update_ReportGenerator(self.checkbox_ActivateReportGenerator))
        # check box - Store Image
        self.checkbox_storeimage = QCheckBox("Store Image")
        self.checkbox_storeimage.setFont(QFont('Arial', 10))
        self.checkbox_storeimage.stateChanged.connect(lambda: self.update_store_images(self.checkbox_storeimage))

        # self.checkbox_evaluation = QCheckBox("Evaluate_Angle")
        # self.checkbox_evaluation.setFont(QFont('Arial', 10))
        # self.checkbox_evaluation.stateChanged.connect(lambda: self.evaluation(self.checkbox_evaluation))
        ### do nothing
        vbox.addWidget(self.image_label)
        #vbox.addWidget(self.fractureSince_textLabel)
        #vbox.addWidget(self.input_box)
        #vbox.addWidget(self.lbl_select_body_part)
        #vbox.addWidget(self.dd_select_body_part)
        #vbox.addWidget(self.lbl_name)
        #vbox.addWidget(self.input_name)
        vbox.addWidget(self.checkbox_storeimage)
        vbox.addWidget(self.checkbox_startstop)
        vbox.addWidget(self.checkbox_Supination_Pronation)
        vbox.addWidget(self.checkbox_evaluation)
        # vbox.addWidget(self.checkbox_ActivateReportGenerator)
        self.NOP = QLabel("TBD")
        self.NOP.setFont(QFont('Arial', 10))
        vbox.addWidget(self.NOP)
        self.NOP = QLabel("TBD")
        self.NOP.setFont(QFont('Arial', 10))
        vbox.addWidget(self.NOP)
        self.NOP = QLabel("TBD")
        self.NOP.setFont(QFont('Arial', 10))
        vbox.addWidget(self.NOP)
        self.setLayout(vbox)
        #self.setLayout(Hbox)
        self.imageutil = ImageConverter()
        # create the video capture thread
        self.thread = video_thread
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        #self.thread.change_pixmap_signal.connect(self.)
        # start the thread
        self.thread.start()
        #self.evaluation(self.checkbox_evaluation)

    def update_start_stop(self, checbox):
        if checbox.isChecked():
            print("Selected : " + str(checbox.text()))
            self.checkbox_startstop.setText("Stop")
        else:
            print("Selected : " + str(checbox.text()))
            self.checkbox_startstop.setText("Start")
        print("Sambath")
        self.thread.on_update_start_stop(checbox.isChecked())
        print(str(checbox.text()))
        print("sambath end")

    def update_Supination_Pronation(self, checbox):
        if checbox.isChecked():
            print("Selected : " + str(checbox.text()))
            self.checkbox_Supination_Pronation.setText("Supination")
        else:
            print("Selected : " + str(checbox.text()))
            self.checkbox_Supination_Pronation.setText("Pronation")
            print("Sambath")
        self.thread.on_Supination_Pronation(checbox.isChecked())
        print(str(checbox.text()))
        print("sambath end")
    def update_ReportGenerator(self, checbox):
        if checbox.isChecked():
            print("Selected : " + str(checbox.text()))
            self.checkbox_ActivateReportGenerator.setText("DeActivateReportGeneration")
        else:
            print("Selected : " + str(checbox.text()))
            self.checkbox_ActivateReportGenerator.setText("ActivateReportGeneration")
        print("Sambath")
        print(str(checbox.text()))
        print("sambath end")

    def update_store_images(self, checbox):
        if checbox.isChecked():
            self.is_store_images = True
        else:
            self.is_store_images = False

    def evaluation(self, checbox):
        self.is_evaluation = True
        if checbox.isChecked():
            self.is_evaluation = True
        else:
            self.is_evaluation = False
        #self.thread.on_update_Evaluate(checbox.isChecked())
        self.thread.on_update_Evaluate(checbox.isChecked())
    def showSelectedBodyPart(self, text):
        print("Selected Body Part is : " + text)
        self.thread.on_update_body_part_selection(text)

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        try:
            rgb_img, qt_img = self.imageutil.convert_cv_2_rgb_qt(cv_img)
            self.image_label.setPixmap(qt_img)
            # send to ML algo
            # save image if store images is True
            self.is_store_images =False
            if self.is_store_images == True:
                self.now = datetime.now()
                self.filename = self.input_name.text() + self.now.strftime("%d%m%Y%H%M%S%f") + '.jpg'
                print("Store Image: " + self.filename)
                if not path.exists(OUTPUT_DIR):
                    os.mkdir(OUTPUT_DIR)
                cv2.imwrite(os.path.join(OUTPUT_DIR, self.filename), rgb_img)
        except:
            print("An Error occured in Video Processing Thread Slot ")

class FourthTab (QWidget):
    def __init__(self,video_thread):
        super().__init__()
        #layout = QVBoxLayout()
        layout = QGridLayout()
        intData=0
        #global evaluatedValue
        self.FOE = QLabel("Flexion of Elbow/Knee: ")
        self.FOE.setStyleSheet("color: black")
        self.FOE.setFont(QFont('Arial', 15))

        self.OBG = QLabel("Observation (Deg): ")
        self.OBG.setFont(QFont('Arial', 15))
        self.OBG.setStyleSheet("color: gray")
        self.OBG_Edit = QLineEdit()
        self.OBG_Edit.setText(str(evaluatedValue))

        self.EXP_Fle = QLabel("Expectation (Deg): ")
        self.EXP_Fle.setFont(QFont('Arial', 15))
        self.EXP_Fle.setStyleSheet("color: gray")
        self.EXP_Edit_Fle = QLineEdit()
        self.EXP_Edit_Fle.setText(str(evaluatedValue))
        #
        self.Compute1 = QPushButton("Compute Flexion")
        self.Compute1.setFont(QFont('Arial', 12))
        self.Compute1.setStyleSheet("background-color: lightgray")
        self.setLayout(layout)
        self.thread = video_thread
        self.Compute1.clicked.connect(self.Compute_Flexion)

        self.LAG_Fle = QLabel("Lag (Deg): ")
        self.LAG_Fle.setFont(QFont('Arial', 15))
        self.LAG_Fle.setStyleSheet("color: gray")
        self.LAG_Edit_Fle = QLineEdit()
        self.LAG_Edit_Fle.setText(str(evaluatedValue))
        layout.addWidget(self.FOE)
        layout.addWidget(self.OBG)
        layout.addWidget(self.OBG_Edit)
        layout.addWidget(self.EXP_Fle)
        layout.addWidget(self.EXP_Edit_Fle)
        layout.addWidget(self.Compute1)
        layout.addWidget(self.LAG_Fle)
        layout.addWidget(self.LAG_Edit_Fle)

        self.EOE_E = QLabel("Extension of Elbow/Knee: ")
        self.EOE_E.setFont(QFont('Arial', 15))
        self.EOE_E.setStyleSheet("color: black")

        self.OBG_E = QLabel("Observation (Deg): ")
        self.OBG_E.setFont(QFont('Arial', 15))
        self.OBG_E.setStyleSheet("color: gray")
        self.OBG_E_Edit1 = QLineEdit()
        self.OBG_E_Edit1.setText(str(evaluatedValue))

        self.EXP_Ext = QLabel("Expectation (Deg): ")
        self.EXP_Ext.setFont(QFont('Arial', 15))
        self.EXP_Ext.setStyleSheet("color: gray")
        self.EXP_Edit_Ext = QLineEdit()
        self.EXP_Edit_Ext.setText(str(evaluatedValue))
        #
        self.Compute2 = QPushButton("Compute Extension")
        self.Compute2.setFont(QFont('Arial', 12))
        self.Compute2.setStyleSheet("background-color: lightgray")
        self.setLayout(layout)
        self.thread = video_thread
        self.Compute2.clicked.connect(self.Compute_Extension)

        self.LAG_Ext = QLabel("Lag (Deg): ")
        self.LAG_Ext.setFont(QFont('Arial', 15))
        self.LAG_Ext.setStyleSheet("color: gray")

        self.LAG_Edit_Ext = QLineEdit()
        self.LAG_Edit_Ext.setText(str(evaluatedValue))
        layout.addWidget(self.EOE_E)
        layout.addWidget(self.OBG_E)
        layout.addWidget(self.OBG_E_Edit1)
        layout.addWidget(self.EXP_Ext)
        layout.addWidget(self.EXP_Edit_Ext)
        layout.addWidget(self.Compute2)
        layout.addWidget(self.LAG_Ext)
        layout.addWidget(self.LAG_Edit_Ext)

########################################################################################################################
       # 5th Tab
########################################################################################################################

        self.save = QPushButton("GENERATE REPORT")
        self.save.setFont(QFont('Arial', 12))
        self.save.setStyleSheet("background-color: lightgreen")
        self.save.clicked.connect(self.saveInput)
        #layout.addWidget(self.save, 0, 0)
        layout.addWidget(self.save,17,0)
        self.setLayout(layout)
        self.thread = video_thread
        # self.SecondTabValue()
        # self.firstTabValue()
        self.thread.change_setDATA.connect(self.firstTabValue)
        print("thread.change_setDATA.connect(self.firstTabValue)")
        self.thread.change_setDATA_2.connect(self.SecondTabValue)
        self.thread.change_elbow_eval_values_signal.connect(self.update_elbow_evaluation_values)
        # self.SecondTabValue(self)
        # self.firstTabValue(self)
        #self.thread.change_setDATA.connect(self.firstTabValue)


    def update_elbow_evaluation_values(self, values):
        #print("Tab 4 - received elbow eval values dict - ")
        #print(values)
        f_val_Min = values["f_min"]
        f_val_Max = values["f_max"]
        #print(f_val_Max)
        #print(f_val_Min)
        self.OBG_Edit.setText(f_val_Min)
        self.OBG_E_Edit1.setText(f_val_Max)

    def Compute_Flexion(self):
        #print("Compute_Flexion")
        #print(self.OBG_Edit.text())
        obs_Deg= self.OBG_Edit.text()
        obs_Deg = int(obs_Deg)

        exp_Deg=self.EXP_Edit_Fle.text()
        #print(self.EXP_Edit_Fle.text())
        exp_Deg = int(exp_Deg)
        lag_Deg = obs_Deg-exp_Deg
        lag_Deg = str(lag_Deg)
        self.LAG_Edit_Fle.setText(lag_Deg)

    def Compute_Extension(self):
        #print("Compute_Extension")
        #print(self.OBG_E_Edit1.text())
        obs_Deg= self.OBG_E_Edit1.text()
        obs_Deg = int(obs_Deg)
        exp_Deg= self.EXP_Edit_Ext.text()
        exp_Deg = int(exp_Deg)
        lag_Deg = obs_Deg-exp_Deg
        lag_Deg = str(lag_Deg)
        self.LAG_Edit_Ext.setText(lag_Deg)


    def SecondTabValue(self, Text_SecondTab):
        print("Text_SecondTab for print")
        self.Text_SecondTab = Text_SecondTab

        self.DOF_1               = self.Text_SecondTab["DOF"]
        self.DOF_Edit_1          = self.Text_SecondTab["DOF_Edit"]

        self.SOF_1               = self.Text_SecondTab["SOF"]
        self.SOF_Edit_Selected_1 = self.Text_SecondTab["SOF_Edit"]

        self.TOF_1               = self.Text_SecondTab["TOF"]
        self.Selected_TOF_Edit_1 = self.Text_SecondTab["TOF_Edit"]

        self.Complication_1      = self.Text_SecondTab["Complication"]
        self.Selected_Complication_Edit_1 =self.Text_SecondTab["Complication_Edit"]

        self.TOC                 = self.Text_SecondTab["TOC"]
        self.Selected_TOC_Edit_1 = self.Text_SecondTab["TOC_Edit"]

        self.DOT_1               = self.Text_SecondTab["DOT"]
        self.DOT_Edit_1          = self.Text_SecondTab["DOT_Edit"]

        self.TOT_1                = self.Text_SecondTab["TOT"]
        self.TOT_Edit_1           = self.Text_SecondTab["TOT_Edit"]

        self.DOTFU_1              = self.Text_SecondTab["DOTFU"]
        self.DOTFU_Edit_1         = self.Text_SecondTab["DOTFU_Edit"]

        self.CFU_1                = self.Text_SecondTab["CFU"]
        self.CFU_Edit_1           = self.Text_SecondTab["CFU_Edit"]

    def firstTabValue(self,Text_FirstTab):
        print("first field - data")
        self.Text_FirstTab = Text_FirstTab
        self.Patient_ID_1     = self.Text_FirstTab["Patient_ID"]
        self.PatientID_Edit_1 = self.Text_FirstTab["PatientID_Edit"]
        self.DOR_1            = self.Text_FirstTab["DOR"]
        self.DOR_Edit_1       = self.Text_FirstTab["DOR_Edit"]
        self.F_Name_1         = self.Text_FirstTab["F_Name"]
        self.F_Name_Edit_1    = self.Text_FirstTab["F_Name_Edit"]
        self.L_Name_1         = self.Text_FirstTab["L_Name"]
        self.L_Name_Edit_1    = self.Text_FirstTab["L_Name_Edit"]
        self.DOB_1            = self.Text_FirstTab["DOB"]
        self.DOB_Edit_1       = self.Text_FirstTab["DOB_Edit"]
        self.Gender_1         = self.Text_FirstTab["Gender"]
        self.Gender_Edit_1    = self.Text_FirstTab["Gender_Edit"]
        self.Address_1        = self.Text_FirstTab["Address"]
        self.Address_Edit_1   = self.Text_FirstTab["Address_Edit"]
        self.Phone_1         = self.Text_FirstTab["Phone"]
        self.Phone_Edit_1     = self.Text_FirstTab["Phone_Edit"]
        self.E_Mail_1         = self.Text_FirstTab["E_Mail"]
        self.E_Mail_Edit_1    = self.Text_FirstTab["E_Mail_Edit"]
        #return self.firstTabValue
    def saveInput(self):
        print("file creation")
        #self.dialog = FirstTab()
        name = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '/', '.txt')[0]
        file = open(name, 'w')
        file.write("Patient Information"+"\n")
        file.write(self.Patient_ID_1+":")
        file.write(self.PatientID_Edit_1+"\n")
        file.write(self.DOR_1+":")
        file.write(self.DOR_Edit_1+"\n")
        file.write(self.F_Name_1+":")
        file.write(self.F_Name_Edit_1+"\n")
        file.write(self.L_Name_1+":")
        file.write(self.L_Name_Edit_1+"\n")
        file.write(self.DOB_1+":")
        file.write(self.DOB_Edit_1+"\n")
        file.write(self.Gender_1+":")
        file.write(self.Gender_Edit_1+"\n")
        file.write(self.Address_1+":")
        file.write(self.Address_Edit_1+"\n")
        file.write(self.Phone_1+":")
        file.write(self.Phone_Edit_1+"\n")
        file.write(self.E_Mail_1+":")
        file.write(self.E_Mail_Edit_1+"\n")
        file.write("Tretment Information" + "\n")
        file.write(self.DOF_1 + ":")
        file.write(self.DOF_Edit_1 + "\n")
        file.write(self.SOF_1 + ":")
        file.write(self.SOF_Edit_Selected_1 + "\n")
        file.write(self.TOF_1 + ":")
        file.write(self.Selected_TOF_Edit_1 + "\n")
        file.write(self.Complication_1 + ":")
        file.write(self.Selected_Complication_Edit_1 + "\n")
        file.write(self.TOC + ":")
        file.write(self.Selected_TOC_Edit_1 + "\n")
        file.write(self.DOT_1 + ":")
        file.write(self.DOT_Edit_1 + "\n")
        file.write(self.TOT_1 + ":")
        file.write(self.TOT_Edit_1 + "\n")
        file.write(self.DOTFU_1 + ":")
        file.write(self.DOTFU_Edit_1 + "\n")
        file.write(self.CFU_1 + ":")
        file.write(self.CFU_Edit_1 + "\n")
        file.write("Flexion Extension Measurement Result" + "\n")

        file.write(self.FOE.text()+ "\n")
        file.write(self.OBG.text()+ ":")
        file.write(self.OBG_Edit.text()+ "\n")
        file.write(self.EXP_Fle.text()+ ":")
        file.write(self.EXP_Edit_Fle.text()+ "\n")
        file.write(self.LAG_Fle.text()+ ":")
        file.write(self.LAG_Edit_Fle.text()+ "\n")
        file.write(self.EOE_E.text()+ "\n")
        file.write(self.OBG_E.text()+ ":")
        file.write(self.OBG_E_Edit1.text()+ "\n")
        file.write(self.EXP_Ext.text()+ ":")
        file.write(self.EXP_Edit_Ext.text()+ "\n")
        file.write(self.LAG_Ext.text()+ ":")
        file.write(self.LAG_Edit_Ext.text()+ "\n")

        print("file about to close")
        file.close()

class FifthTab (QWidget):
    def __init__(self,video_thread):
        super().__init__()
        layout = QVBoxLayout()
        self.FOE = QLabel("Supination: ")
        self.FOE.setFont(QFont('Arial', 15))
        self.FOE.setStyleSheet("color: black")

        self.OBG = QLabel("Observation (Deg): ")
        self.OBG.setStyleSheet("color: gray")
        self.OBG.setFont(QFont('Arial', 15))
        self.OBG_Edit2 = QLineEdit()
        #self.OBG_Edit2.setFixedWidth(200)
        self.OBG_Edit2.setText(str(evaluatedValue))

        self.EXP_Supi = QLabel("Expectation (Deg): ")
        self.EXP_Supi.setStyleSheet("color: gray")
        self.EXP_Supi.setFont(QFont('Arial', 15))
        self.EXP_Edit_Supi = QLineEdit()
        self.EXP_Edit_Supi.setText(str(evaluatedValue))

        self.LAG_Supi = QLabel("Lag (Deg): ")
        self.LAG_Supi.setFont(QFont('Arial', 15))
        self.LAG_Supi.setStyleSheet("color: gray")

        self.LAG_Edit_Supi = QLineEdit()
        #
        self.Compute1 = QPushButton("Compute Supination")
        self.Compute1.setFont(QFont('Arial', 12))
        self.Compute1.setStyleSheet("background-color: lightgray")
        self.setLayout(layout)
        self.thread = video_thread
        self.Compute1.clicked.connect(self.Compute_Flection)
        #
        layout.addWidget(self.FOE)
        layout.addWidget(self.OBG)
        layout.addWidget(self.OBG_Edit2)
        layout.addWidget(self.EXP_Supi)
        layout.addWidget(self.EXP_Edit_Supi)
        layout.addWidget(self.LAG_Supi )
        layout.addWidget(self.Compute1)
        layout.addWidget(self.LAG_Edit_Supi)

        self.EOE = QLabel("Pronation: ")
        self.EOE.setFont(QFont('Arial', 15))
        self.EOE.setStyleSheet("color: black")

        self.OBG_pro = QLabel("Observation (Deg): ")
        self.OBG_pro.setFont(QFont('Arial', 15))
        self.OBG_pro.setStyleSheet("color: gray")
        self.OBG_Edit3 = QLineEdit()
        self.OBG_Edit3.setText(str(evaluatedValue))

        self.EXP_Pro = QLabel("Expectation (Deg): ")
        self.EXP_Pro.setFont(QFont('Arial', 15))
        self.EXP_Pro.setStyleSheet("color: gray")
        self.EXP_Edit_Pro = QLineEdit()
        self.EXP_Edit_Pro.setText(str(evaluatedValue))
        self.Compute2 = QPushButton("Compute Pronation")
        self.Compute2.setFont(QFont('Arial', 12))
        self.Compute2.setStyleSheet("background-color: lightgray")
        self.setLayout(layout)
        self.thread = video_thread
        self.Compute2.clicked.connect(self.Compute_Pronation)

        self.LAG_Pro = QLabel("Lag (Deg): ")
        self.LAG_Pro.setFont(QFont('Arial', 15))
        self.LAG_Pro.setStyleSheet("color: gray")
        self.LAG_Edit_Pro = QLineEdit()
        layout.addWidget(self.EOE)
        layout.addWidget(self.OBG_pro)
        layout.addWidget(self.OBG_Edit3)
        layout.addWidget(self.EXP_Pro)
        layout.addWidget(self.EXP_Edit_Pro)
        layout.addWidget(self.Compute2)
        layout.addWidget(self.LAG_Pro)
        layout.addWidget(self.LAG_Edit_Pro)

        self.save = QPushButton("GENERATE REPORT")
        self.save.setFont(QFont('Arial', 12))
        self.save.setStyleSheet("background-color: lightgreen")
        self.save.clicked.connect(self.saveInput)
        #layout.addWidget(self.save, 0, 0)
        layout.addWidget(self.save)

        self.setLayout(layout)
        self.thread = video_thread
        self.thread.change_gunshot_eval_values_signal.connect(self.update_finger_evaluation_values)
        self.thread.change_setDATA.connect(self.firstTabValue)
        self.thread.change_setDATA_2.connect(self.SecondTabValue)
    def update_finger_evaluation_values(self, values):
        #print("Tab 5 - received finger eval values dict - ")
        # self.xxxxx.setText(str(values["sample_key"]))
        f_val_Max = values["f_max"]
        #result = VideoProcessingThread.on_Supination_Pronation()
        print(supinationPronation)
        #self.OBG_Edit2.setText(f_val_Max)
        if(supinationPronation == True):
            self.OBG_Edit2.setText(f_val_Max)
        else:
            self.OBG_Edit3.setText(f_val_Max)
        print(f_val_Max)

    def Compute_Flection(self):
        print("save - emit")
        print("")
        print(self.OBG_Edit2.text())
        obs_Deg= self.OBG_Edit2.text()
        obs_Deg = int(obs_Deg)
        exp_Deg=self.EXP_Edit_Supi.text()
        exp_Deg = int(exp_Deg)
        lag_Deg = obs_Deg-exp_Deg
        lag_Deg = str(lag_Deg)
        self.LAG_Edit_Supi.setText(lag_Deg)

    def Compute_Pronation(self):
        print("save - emit")
        print(self.OBG_Edit3.text())
        obs_Deg= self.OBG_Edit3.text()
        obs_Deg = int(obs_Deg)

        exp_Deg=self.EXP_Edit_Pro.text()
        exp_Deg = int(exp_Deg)
        lag_Deg = obs_Deg-exp_Deg
        lag_Deg = str(lag_Deg)
        self.LAG_Edit_Pro.setText(lag_Deg)

    def SecondTabValue(self, Text_SecondTab):
        print("Text_SecondTab for fith tab")
        self.Text_SecondTab = Text_SecondTab

        self.DOF_1               = self.Text_SecondTab["DOF"]
        self.DOF_Edit_1          = self.Text_SecondTab["DOF_Edit"]

        self.SOF_1               = self.Text_SecondTab["SOF"]
        self.SOF_Edit_Selected_1 = self.Text_SecondTab["SOF_Edit"]

        self.TOF_1               = self.Text_SecondTab["TOF"]
        self.Selected_TOF_Edit_1 = self.Text_SecondTab["TOF_Edit"]

        self.Complication_1      = self.Text_SecondTab["Complication"]
        self.Selected_Complication_Edit_1 =self.Text_SecondTab["Complication_Edit"]

        self.TOC                 = self.Text_SecondTab["TOC"]
        self.Selected_TOC_Edit_1 = self.Text_SecondTab["TOC_Edit"]

        self.DOT_1               = self.Text_SecondTab["DOT"]
        self.DOT_Edit_1          = self.Text_SecondTab["DOT_Edit"]

        self.TOT_1                = self.Text_SecondTab["TOT"]
        self.TOT_Edit_1           = self.Text_SecondTab["TOT_Edit"]

        self.DOTFU_1              = self.Text_SecondTab["DOTFU"]
        self.DOTFU_Edit_1         = self.Text_SecondTab["DOTFU_Edit"]

        self.CFU_1                = self.Text_SecondTab["CFU"]
        self.CFU_Edit_1           = self.Text_SecondTab["CFU_Edit"]
        print("Text_SecondTab for fith tab - ends")
    def firstTabValue(self,Text_FirstTab):
        print("first field - for 5th tab")
        self.Text_FirstTab = Text_FirstTab
        self.Patient_ID_1     = self.Text_FirstTab["Patient_ID"]
        self.PatientID_Edit_1 = self.Text_FirstTab["PatientID_Edit"]
        self.DOR_1            = self.Text_FirstTab["DOR"]
        self.DOR_Edit_1       = self.Text_FirstTab["DOR_Edit"]
        self.F_Name_1         = self.Text_FirstTab["F_Name"]
        self.F_Name_Edit_1    = self.Text_FirstTab["F_Name_Edit"]
        self.L_Name_1         = self.Text_FirstTab["L_Name"]
        self.L_Name_Edit_1    = self.Text_FirstTab["L_Name_Edit"]
        self.DOB_1            = self.Text_FirstTab["DOB"]
        self.DOB_Edit_1       = self.Text_FirstTab["DOB_Edit"]
        self.Gender_1         = self.Text_FirstTab["Gender"]
        self.Gender_Edit_1    = self.Text_FirstTab["Gender_Edit"]
        self.Address_1        = self.Text_FirstTab["Address"]
        self.Address_Edit_1   = self.Text_FirstTab["Address_Edit"]
        self.Phone_1         = self.Text_FirstTab["Phone"]
        self.Phone_Edit_1     = self.Text_FirstTab["Phone_Edit"]
        self.E_Mail_1         = self.Text_FirstTab["E_Mail"]
        self.E_Mail_Edit_1    = self.Text_FirstTab["E_Mail_Edit"]
        #return self.firstTabValue
    def saveInput(self):
        print("file creation")
        #self.dialog = FirstTab()
        name = QtWidgets.QFileDialog.getSaveFileName(self, "Save File", '/', '.txt')[0]
        file = open(name, 'w')
        file.write("Patient Information"+"\n")
        file.write(self.Patient_ID_1+":")
        file.write(self.PatientID_Edit_1+"\n")
        file.write(self.DOR_1+":")
        file.write(self.DOR_Edit_1+"\n")
        file.write(self.F_Name_1+":")
        file.write(self.F_Name_Edit_1+"\n")
        file.write(self.L_Name_1+":")
        file.write(self.L_Name_Edit_1+"\n")
        file.write(self.DOB_1+":")
        file.write(self.DOB_Edit_1+"\n")
        file.write(self.Gender_1+":")
        file.write(self.Gender_Edit_1+"\n")
        file.write(self.Address_1+":")
        file.write(self.Address_Edit_1+"\n")
        file.write(self.Phone_1+":")
        file.write(self.Phone_Edit_1+"\n")
        file.write(self.E_Mail_1+":")
        file.write(self.E_Mail_Edit_1+"\n")
        print("Firstpage data is completed")
        file.write("Tretment Information" + "\n")
        file.write(self.DOF_1 + ":")
        file.write(self.DOF_Edit_1 + "\n")
        file.write(self.SOF_1 + ":")
        file.write(self.SOF_Edit_Selected_1 + "\n")
        file.write(self.TOF_1 + ":")
        file.write(self.Selected_TOF_Edit_1 + "\n")
        file.write(self.Complication_1 + ":")
        file.write(self.Selected_Complication_Edit_1 + "\n")
        file.write(self.TOC + ":")
        file.write(self.Selected_TOC_Edit_1 + "\n")
        file.write(self.DOT_1 + ":")
        file.write(self.DOT_Edit_1 + "\n")
        file.write(self.TOT_1 + ":")
        file.write(self.TOT_Edit_1 + "\n")
        file.write(self.DOTFU_1 + ":")
        file.write(self.DOTFU_Edit_1 + "\n")
        file.write(self.CFU_1 + ":")
        file.write(self.CFU_Edit_1 + "\n")
        print("seconpage data is completed")
        file.write("Supination and Pronation Result" + "\n")

        file.write(self.FOE.text() + "\n")
        file.write(self.OBG.text()+ ":")
        print(self.OBG.text())
        file.write(self.OBG_Edit2.text()+ "\n")
        print("333")

        file.write(self.EXP_Supi.text()+ ":")
        print(self.EXP_Supi.text())
        file.write(self.EXP_Edit_Supi.text()+ "\n")
        print(self.EXP_Edit_Supi.text())

        file.write(self.LAG_Supi.text()+ ":")
        print(self.LAG_Supi.text())
        file.write(self.LAG_Edit_Supi.text()+ "\n")
        print(self.LAG_Edit_Supi.text())

        file.write(self.EOE.text()+ "\n")
        print(self.EOE.text())

        file.write(self.OBG_pro.text()+ ":")
        print(self.OBG_pro.text())
        file.write(self.OBG_Edit3.text()+ "\n")

        file.write(self.EXP_Pro.text()+ ":")
        print(self.EXP_Pro.text())
        file.write(self.EXP_Edit_Pro.text()+ "\n")
        print(self.EXP_Edit_Pro.text())

        file.write(self.LAG_Pro.text()+ ":")
        print(self.LAG_Pro.text())
        file.write(self.LAG_Edit_Pro.text()+ "\n")
        print(self.LAG_Edit_Pro.text())

        print("file about to close")
        file.close()

class SixthTab (QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        FOE = QLabel("")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)
        FOE = QLabel("")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)
        FOE = QLabel("")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)

        FOE = QLabel("Recommendation based on the current ")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)
        FOE_1 = QLabel("and artificial intelligence data: ")
        FOE_1.setFont(QFont('Arial', 50))
        layout.addWidget(FOE_1)
        FOE_1 = QLabel("")
        FOE_1.setFont(QFont('Arial', 50))
        layout.addWidget(FOE_1)
        FOE = QLabel("")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)
        FOE = QLabel("")
        FOE.setFont(QFont('Arial',50))
        layout.addWidget(FOE)
        self.setLayout(layout)

class AnotherWindow(QWidget):
    def __init__(self, ):
        super().__init__()
        self.setWindowTitle("DETACTION - Calendar")
        self.setWindowIcon(QIcon("myicon.png"))
        #print("AnotherWindow")
        self.calendar = QCalendarWidget(self)
        #self.calendar.setStyleSheet("background-color : lightgreen;")
        self.calendar.setStyleSheet("color: blue; background-color : lightgreen;")

        date = self.calendar.selectedDate()
        #self.Date = date.currentDate()
        self.Date = date.getDate()
        print("another window1")
        print(self.Date)
        print(date)
        print("another window2")
        layout = QVBoxLayout()
        self.label = QLabel("Another Window % d" % randint(0,100))
        layout.addWidget(self.calendar)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def getCalender(self):
        return self.calendar

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet('* {color: yellow;}')
    #app.setStyleSheet("background-color: green;")
    a = App()
    qMainWindow = QMainWindow()
    desktop = QApplication.desktop()
    screenRect = desktop.screenGeometry()
    height = screenRect.height()
    print(height)
    width = screenRect.width()
    print(width)
    #qMainWindow.resize(1400, 800)
    qMainWindow.resize(width, height)
    qMainWindow.setWindowTitle("DETACTION - AI Based ORTHOPEDIC ASSIST")
    qMainWindow.setWindowIcon(QIcon("myicon.png"))
    qMainWindow.setCentralWidget(a)
    qMainWindow.show()
    sys.exit(app.exec_())
