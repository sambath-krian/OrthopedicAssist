# BETWEEN FINGER ANGLE

import mediapipe as mp
import cv2
import numpy as np


from matplotlib import pyplot as plt
#os.mkdir('Output Images')
#joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]

#joint_list[3]
# joint_list[1]
first = 0
a = []

a_list =[]
index = 0
class Angle_Calc_Gunshot:

    def __init__(self):
        self.mp_hand = mp.solutions.hands.Hands(min_detection_confidence=0.9, min_tracking_confidence=0.1)
        self.isStartSet = False
        #self.flex_min_Value = 0
        self.values_dict = {}
        self.flex_max_Value = 0

        #self.joint_list = [[8, 0, 4], [12, 9, 8], [16, 13, 12], [12, 11, 10]]
    def get_eval_values(self):
        #print("get_eval_values - Pronation/supination")
        self.values_dict["f_max"] = str((self.flex_max_Value))
        # print("xxxx")
        # print(self.values_dict["f_max"])
        return self.values_dict

    def draw_finger_angles(self, image, results, Sup):
        global first
        global index
        global a
        self.isevaluation_Sup = Sup
        #print("Gunshot call 11")
        # Loop through hands
        for self.hand in self.results.multi_hand_landmarks:
            ##print (self.results.multi_hand_landmarks)
            #print("dada")
            #print(self.hand)
            if (self.isStartSet == False):
                #self.che
                print("problem in firstcall")
                a = np.array([self.hand.landmark[4].x, self.hand.landmark[4].y])  # First coord
                first = 1
            a = a
            b = np.array([self.hand.landmark[9].x, self.hand.landmark[9].y])   # Second coord
            c = np.array([self.hand.landmark[4].x, self.hand.landmark[4].y])   # Third coord
            
        # for self.hand in self.results.multi_hand_landmarks:
        #     # Loop through joint sets
        #     for self.joint in self.joint_list:
        #         print("test vec. points")
        #         print(self.joint)
        #         print (self.joint_list)
        #         a = np.array([self.hand.landmark[self.joint[0]].x, self.hand.landmark[self.joint[0]].y])  # First coord
        #         b = np.array([self.hand.landmark[self.joint[1]].x, self.hand.landmark[self.joint[1]].y])  # Second coord
        #         c = np.array([self.hand.landmark[self.joint[2]].x, self.hand.landmark[self.joint[2]].y])  # Third coord

            self.radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
            self.angle = np.abs(self.radians * 180.0 / np.pi)

            if self.angle > 180.0:
                self.angle = 360 - self.angle
            #print("ANGLE")
            #print(self.angle)
            if(self.isStartSet == False):
                index = 0
                a_list.clear()
                a_list.clear()
            else:
                index = index + 1
            a_list.insert(index, round(self.angle))
            #print(a_list)
            self.flex_max_Value = max(a_list)
            # if(self.isevaluation_Sup == True):
            # # self.flex_min_Value = min(a_list)
            #     self.flex_max_Value = max(a_list)
            # # a_list.clear()
            #     print(self.min_Value)
            #     print("isevaluation_Sup")
            # # print("ffffffffffffffffffffffffffffffffffff")
            # else:
            #     pass

            cv2.putText(image, str(round(self.angle,0)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            return image

    def process(self, frame, isStartSet, eval_Supination):

        width = int(frame.shape[1] * 1)
        height = int(frame.shape[0] * 1)
        dim = (width, height)
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # BGR 2 RGBframe
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Set parameters
        self.isStartSet = isStartSet
        self.isevaluation_Sup = eval_Supination
        #self.isevaluation_Pro = eval_Pronation

        # Flip on horizontal
        self.image = cv2.flip(self.image, 1)

        # Set flag
        self.image.flags.writeable = False

        # Detections
        self.results = self.mp_hand.process(self.image)

        # Set flag to true
        self.image.flags.writeable = True

        # RGB 2 BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        scale_percent = 60
        # width = int(self.image.shape[1] * scale_percent / 100)
        # height = int(self.image.shape[0] * scale_percent / 100)
        # width = int(self.image.shape[1] * 2)
        # height = int(self.image.shape[0] * 2)
        # dim = (width, height)
        # self.imag = cv2.resize(self.image, dim, interpolation=cv2.INTER_AREA)
        # Detections
        #print(self.results)
        #print("11")
        # Rendering results
        if self.results.multi_hand_landmarks:
            print("Gunshot call 22")
            for self.num, self.hand in enumerate(self.results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(self.image, self.hand, mp.solutions.hands.HAND_CONNECTIONS,
                                          mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                          mp.solutions.drawing_utils.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                          )

                # Render left or right detection
                #if get_label(num, hand, results):
                    #text, coord = get_label(num, hand, results)
                    #cv2.putText(image, text, coord, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw angles to image from joint list
            self.image = self.draw_finger_angles(self.image, self.results, self.isevaluation_Sup)

        # Save our image
        # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        # cv2.imshow('Hand Tracking', image)
        return self.image
