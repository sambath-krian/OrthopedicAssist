# BETWEEN FINGER ANGLE

import mediapipe as mp
import cv2
import numpy as np


from matplotlib import pyplot as plt
#os.mkdir('Output Images')
#joint_list = [[8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]

#joint_list[3]
# joint_list[1]

class Angle_Calculator_finger:

    def __init__(self):
        self.mp_hand = mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.2)
        #self testing
        self.joint_list = [[8, 0, 4], [12, 9, 8], [16, 13, 12], [12, 11, 10]]
        self.eval_values = {}

    def get_eval_values(self):
        self.eval_values["sample_key"] = 0
        return self.eval_values

    def draw_finger_angles(self, image, results):
        if(1==self.fingerAngSelect):
            self.joint_list = [[4, 0, 8]]
        elif(2==self.fingerAngSelect):
            self.joint_list = [[8, 9, 12]]
        elif(3==self.fingerAngSelect):
            self.joint_list = [[12, 13, 16]]
        elif(4==self.fingerAngSelect):
            self.joint_list = [[16, 17, 20]]
        else:
            pass
        # Loop through hands
        print("finger call 11")
        for self.hand in self.results.multi_hand_landmarks:
            # Loop through joint sets
            for self.joint in self.joint_list:
                a = np.array([self.hand.landmark[self.joint[0]].x, self.hand.landmark[self.joint[0]].y])  # First coord
                b = np.array([self.hand.landmark[self.joint[1]].x, self.hand.landmark[self.joint[1]].y])  # Second coord
                c = np.array([self.hand.landmark[self.joint[2]].x, self.hand.landmark[self.joint[2]].y])  # Third coord

                self.radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                self.angle = np.abs(self.radians * 180.0 / np.pi)

                if self.angle > 180.0:
                    self.angle = 360 - self.angle

                cv2.putText(image, str(round(self.angle, 0)), tuple(np.multiply(b, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def process(self, frame, fingerAngSelect):
        # BGR 2 RGB
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        self.image = cv2.flip(self.image, 1)

        # Set flag
        self.image.flags.writeable = False
        self.fingerAngSelect = fingerAngSelect

        # Detections
        self.results = self.mp_hand.process(self.image)

        # Set flag to true
        self.image.flags.writeable = True

        # RGB 2 BGR
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        # Detections
        print(self.results)
        print("11")
        # Rendering results
        if self.results.multi_hand_landmarks:
            print("finger call 22")
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
            self.draw_finger_angles(self.image, self.results)

        # Save our image
        # cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        # cv2.imshow('Hand Tracking', image)
        return self.image
