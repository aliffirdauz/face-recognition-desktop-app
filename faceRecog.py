import tkinter as tk
from tkinter import simpledialog
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage
import cv2
import time
import numpy as np
import face_recognition
import imutils
import pickle
import pyshine as ps
from tkinter import messagebox


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(498, 522)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("images/H.png"))
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.verticalSlider = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider.setObjectName("verticalSlider")
        self.gridLayout.addWidget(self.verticalSlider, 0, 0, 1, 1)
        self.verticalSlider_2 = QtWidgets.QSlider(self.centralwidget)
        self.verticalSlider_2.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_2.setObjectName("verticalSlider_2")
        self.gridLayout.addWidget(self.verticalSlider_2, 0, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.gridLayout)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_2.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 1, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(313, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 1, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.verticalSlider.valueChanged['int'].connect(self.brightness_value)
        self.verticalSlider_2.valueChanged['int'].connect(self.blur_value)
        self.pushButton_3.clicked.connect(self.addImage)
        self.pushButton_2.clicked.connect(self.loadImage)
        self.pushButton.clicked.connect(self.savePhoto)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.filename = 'Snapshot ' + str(
            time.strftime("%Y-%b-%d at %H.%M.%S %p")) + '.png'
        self.tmp = None
        self.brightness_value_now = 0
        self.blur_value_now = 0
        self.fps = 0
        self.started = False

    def addImage(self):
        ROOT = tk.Tk()
        ROOT.withdraw()
        name = simpledialog.askstring(title="Add Face",
                                  prompt="What's your Name?:")
        ref_id = simpledialog.askstring(title="Add ID",
                                  prompt="What's your ID?:")
        if name=='' or ref_id=='' :
            messagebox.showerror("ERROR","Terdapat data kosong!")
        else :
            try:
                f = open("ref_name.pkl", "rb")

                ref_dictt = pickle.load(f)
                f.close()
            except:
                ref_dictt = {}
            ref_dictt[ref_id] = name

            f = open("ref_name.pkl", "wb")
            pickle.dump(ref_dictt, f)
            f.close()

            try:
                f = open("ref_embed.pkl", "rb")

                embed_dictt = pickle.load(f)
                f.close()
            except:
                embed_dictt = {}

            for i in range(10):
                key = cv2.waitKey(1)
                webcam = cv2.VideoCapture(0)
                while True:

                    check, frame = webcam.read()

                    cv2.imshow("Capturing", frame)
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]

                    key = cv2.waitKey(1)

                    if key == ord('s'):
                        face_locations = face_recognition.face_locations(rgb_small_frame)
                        if face_locations != []:
                            face_encoding = face_recognition.face_encodings(frame)[0]
                            if ref_id in embed_dictt:
                                embed_dictt[ref_id] += [face_encoding]
                            else:
                                embed_dictt[ref_id] = [face_encoding]
                            webcam.release()
                            cv2.waitKey(1)
                            cv2.destroyAllWindows()
                            break
                    elif key == ord('q'):
                        print("Turning off camera.")
                        webcam.release()
                        print("Camera off.")
                        print("Program ended.")
                        cv2.destroyAllWindows()
                        break

            f = open("ref_embed.pkl", "wb")
            pickle.dump(embed_dictt, f)
            f.close()

    def loadImage(self):
        if self.started:
            self.started = False
            self.pushButton_2.setText('Start')
        else:
            self.started = True
            self.pushButton_2.setText('Stop')

        f = open("ref_name.pkl", "rb")
        ref_dictt = pickle.load(f)
        f.close()

        f = open("ref_embed.pkl", "rb")
        embed_dictt = pickle.load(f)
        f.close()

        known_face_encodings = []
        known_face_names = []

        for ref_id, embed_list in embed_dictt.items():
            for my_embed in embed_list:
                known_face_encodings += [my_embed]
                known_face_names += [ref_id]

        vid = cv2.VideoCapture(0)
        cnt = 0
        frames_to_count = 20
        st = 0
        fps = 0

        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while (vid.isOpened()):
            QtWidgets.QApplication.processEvents()
            ret, self.image = vid.read()

            small_frame = cv2.resize(self.image, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            if process_this_frame:

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:

                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)

            process_this_frame = not process_this_frame

            for (top_s, right, bottom, left), name in zip(face_locations, face_names):
                top_s *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(self.image, (left, top_s), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(self.image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(self.image, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            font = cv2.FONT_HERSHEY_DUPLEX

            if cnt == frames_to_count:
                try:
                    print(frames_to_count / (time.time() - st), 'FPS')
                    self.fps = round(frames_to_count / (time.time() - st))

                    st = time.time()
                    cnt = 0
                except:
                    pass

            cnt += 1

            self.update()
            key = cv2.waitKey(1) & 0xFF
            if self.started == False:
                break
                print('Loop break')

    def setPhoto(self, image):
        self.tmp = image
        image = imutils.resize(image, width=640)
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(image))

    def brightness_value(self, value):
        self.brightness_value_now = value
        print('Brightness: ', value)
        self.update()

    def blur_value(self, value):
        self.blur_value_now = value
        print('Blur: ', value)
        self.update()

    def changeBrightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    def changeBlur(self, img, value):
        kernel_size = (value + 1, value + 1)  # +1 is to avoid 0
        img = cv2.blur(img, kernel_size)
        return img

    def update(self):
        img = self.changeBrightness(self.image, self.brightness_value_now)
        img = self.changeBlur(img, self.blur_value_now)

        text = 'FPS: ' + str(self.fps)
        img = ps.putBText(img, text, text_offset_x=20, text_offset_y=30, vspace=20, hspace=10, font_scale=1.0,
                          background_RGB=(10, 20, 222), text_RGB=(255, 255, 255))
        text = str(time.strftime("%H:%M %p"))
        img = ps.putBText(img, text, text_offset_x=self.image.shape[1] - 180, text_offset_y=30, vspace=20, hspace=10,
                          font_scale=1.0, background_RGB=(228, 20, 222), text_RGB=(255, 255, 255))
        text = f"Brightness: {self.brightness_value_now}"
        img = ps.putBText(img, text, text_offset_x=20, text_offset_y=425, vspace=20, hspace=10, font_scale=1.0,
                          background_RGB=(20, 210, 4), text_RGB=(255, 255, 255))
        text = f"Blur: {self.blur_value_now}"
        img = ps.putBText(img, text, text_offset_x=self.image.shape[1] - 180, text_offset_y=425, vspace=20, hspace=10,
                          font_scale=1.0, background_RGB=(210, 20, 4), text_RGB=(255, 255, 255))

        self.setPhoto(img)

    def savePhoto(self):
        """ This function will save the image"""
        self.filename = 'Snapshot ' + str(time.strftime("%Y-%b-%d at %H.%M.%S %p")) + '.png'
        cv2.imwrite(self.filename, self.tmp)
        print('Image saved as:', self.filename)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Pengenalan Pola Wajah"))
        self.pushButton_2.setText(_translate("MainWindow", "Start"))
        self.label_2.setText(_translate("MainWindow", "Brightness"))
        self.label_3.setText(_translate("MainWindow", "Blur"))
        self.pushButton.setText(_translate("MainWindow", "Save Result"))
        self.pushButton_3.setText(_translate("MainWindow", "Take Face"))

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())