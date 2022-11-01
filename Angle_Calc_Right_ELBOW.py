# DUMB BELL LIFTING

import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
a_list =[]
index = 0

class Angle_Calc_Right_ELBOW:

    def __init__(self):
        self.mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.flex_min_Value = 0
        self.flex_max_Value = 0
        self.values_dict = {}

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        self.radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        self.angle = np.abs(self.radians * 180.0 / np.pi)

        if self.angle > 180.0:
            self.angle = 360 - self.angle

        return self.angle

    def get_eval_values(self):
        #print("get_eval_values - right elbow")
        #print(type(self.flex_min_Value))
        #self.values_dict["f_min"] = self.flex_min_Value.astype('|S10')
        self.values_dict["f_min"] = str((self.flex_min_Value))
        self.values_dict["f_max"] = str((self.flex_max_Value))
        return self.values_dict

    def process(self, frame, isStartSet, evaluation):
        # Recolor image to RGB
        global index
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Flip on horizontal
        self.image = cv2.flip(self.image, 1)

        self.image.flags.writeable = False
        self.isStartSet = isStartSet
        self.isevaluation = evaluation

        # Make detection
        self.results = self.mp_pose.process(self.image)

        # Recolor back to BGR
        self.image.flags.writeable = True
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            self.landmarks = self.results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                     self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                     self.landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle
            angle = self.calculate_angle(shoulder, elbow, wrist)

            # Visualize angle
            cv2.putText(self.image, str(round(angle, 0)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            cv2.putText(self.image, 'Right Elbow :', (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
            #cv2.putText(self.image, str(round(angle, 0)),
                        # tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        # )
            if (self.isStartSet == False):
                index =  0
                a_list.clear()
            else:
                index = index + 1
            a_list.insert(index,round(angle))
            #print(a_list)
            if(self.isevaluation == True):
                self.flex_min_Value = min(a_list)
                self.flex_max_Value = max(a_list)


                #a_list.clear()
                # print(self.min_Value)
                # print("ffffffffffffffffffffffffffffffffffff")
            else:
                pass
        except:
            pass

        # Render detections
        mp.solutions.drawing_utils.draw_landmarks(self.image, self.results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                  mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        return self.image


