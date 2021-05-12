#!/usr/bin/python3

#ROS imports
import sys
import rospy
import cv2
import time
import mediapipe as mp
import numpy as np
import ast
import threading
from copy import deepcopy

from std_msgs.msg                   import String
from sensor_msgs.msg                import Image
from geometry_msgs.msg              import Point
from cv_bridge                      import CvBridge, CvBridgeError


class PoseEstimator():
    def __init__(self):
        # CV2 Constants
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.DOT_COLOR = self.BLUE
        self.LINE_COLOR = self.BLUE
        
        # Drawing flags
        self.DRAW_LINES = 1
        self.DRAW_POSE_LINES = 0
        self.DRAW_HAND_LINES = 0
        self.DRAW_CIRCLES = 1

        # Estimator constants
        self.NUM_MIDPOINTS = 25
        self.ground_truth = None
        self.midpoints = None
        self.running = False
        self.ARM_THRESHOLD = None
        self.FINGER_THRESHOLD = None
        self.POINTING_THRESHOLD = None

        # Feature detector
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Keypoint data structures
        self.pose_keypoints = []
        self.left_hand_keypoints = []
        self.right_hand_keypoints = []

        # ROS stuffs
        self.bridge = CvBridge()
        self.source_image = None
        self.image_sub = rospy.Subscriber('/cam/image_raw', Image, self.get_image)
        self.pointing_pub = rospy.Publisher('/pose/pointing_at', String, queue_size=10)

    """
    This function is the main entry point to the program. This is where the user decides what to do.
    """
    def menu(self):
        print("================================")
        print("Current Ground Truth:", self.ground_truth)
        print("Exit program: 0")
        print("Run pose estimation: 1")
        print("Create ground truth: 2")
        print("Save ground truth to file: 3")
        print("Load ground truth from file: 4")
        try:
            choice = int(input())
        except ValueError:
            print("Error, please make a choice from the list!")
            return

        if choice == 0:
            sys.exit()
        elif choice == 1:
            self.run_pose_estimation()
        elif choice == 2:
            self.create_ground_truth()
        elif choice == 3:
            if self.ground_truth is not None:
                self.save_ground_truth()
            else:
                print("Error: No ground truth. Create one first.")
        elif choice == 4:
            self.load_ground_truth()
        else:
            print("Error, please make a choice from the list!")

    """
    This function helps the user create the ground truth to use with the pose estimation.
    The user will click on points in the frame and name them in the console.
    Cease the operation by typing q with the camera feed window selected
    params: data - ROS Image message
    """
    ##TODO Refactor video capture to use ROS subscriber 
    def create_ground_truth(self):
        self.ground_truth = {}
        while True:
            cv2.imshow("Camera Feed", self.source_image)
            cv2.setMouseCallback("Camera Feed", self.click_event)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                print("Ground truth created")
                break

    """
    This is a helper function for create_ground_truth, it listens for clicks in the camera feed window
    """
    def click_event(self, event, x, y, flags, params):
        # Checking for left click
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.imshow("Camera Feed", self.source_image)
            x = x//80
            y = y//60
            print(x, y)
            name = input("Enter the object name: ")
            self.ground_truth[(x, y)] = name

    """
    This function saves the ground truth variable, if it exists, to the file specified by the user
    """
    def save_ground_truth(self):
        file_name = input("Enter the ground truth file name to save: ")
        file = open(file_name, 'w+')
        file.write(str(self.ground_truth))
        file.close()
        print("Ground truth saved to:", file_name)

    """
    This function loads the ground truth from the file specified by the user, if it exists
    """
    def load_ground_truth(self):
        file_name = input("Enter the ground truth file name to load: ")
        try:
            file = open(file_name, 'r')
        except FileNotFoundError:
            print("Error:", file_name, "does not exist!")
            return
        str = file.read()
        if str is not '':
            self.ground_truth = ast.literal_eval(str)
        else:
            print("Error: Nothing present in", file_name, "file!")
        print("Ground truth loaded!")

    """
    This funciton checks the coords variable to see if they coorespond to anything in the ground_truth variable
    """
    def check_coords(self):
        while self.running:
            if self.midpoints and self.ground_truth:
                objects = []
                for point in self.midpoints:
                    if point in self.ground_truth:
                        objects.append(self.ground_truth[point])
                obj_set = set(objects)
                if obj_set:
                    print("You are pointing at: ", [item for item in obj_set])
                    self.pointing_pub.publish(String(str([item for item in obj_set])))
            time.sleep(1)

    """
    This is the callback function for the ROS image subscriber
    """
    def get_image(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        self.source_image = deepcopy(img)

    """
    This function performs the pose estimation operations. It reads the camera, makes predictions, and updates internal variables based on the human pose.
    """
    def run_pose_estimation(self):
        # Get image from ROS
        self.running = True
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            self.thread = threading.Thread(target=self.check_coords)
            self.thread.start()
            while True:
                self.image = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)

                # Make predictions of human pose
                self.results = holistic.process(self.image)
                self.get_keypoints()
                self.draw_lines()

                # Check if pointing to a certain place
                self.midpoints = self.determine_pointing()

                # Show image
                self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera Feed", self.image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    self.running = False
                    self.thread.join()
                    cv2.destroyAllWindows()
                    break

    """
    This function will extract the pose keypoints from the results and will determine the pointing measurements for the left arm.
    """
    def get_keypoints(self):
        if self.results.pose_landmarks:
            self.pose_keypoints = []
            for data_point in self.results.pose_landmarks.landmark:
                self.pose_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })
            # Left Arm
            self.l_shoulder = self.pose_keypoints[11]
            self.l_wrist = self.pose_keypoints[15]
            self.l_hip = self.pose_keypoints[23]

            self.left_arm_sX = int(self.l_shoulder['X']*640)
            self.left_arm_sY = int(self.l_shoulder['Y']*480)
            self.left_arm_eX = int(self.l_wrist['X']*640)
            self.left_arm_eY = int(self.l_wrist['Y']*480)

            self.left_arm_length = np.linalg.norm(np.array([self.left_arm_sX,self.left_arm_sY])-np.array([self.left_arm_eX,self.left_arm_eY]))

            # Right Arm
            self.r_shoulder = self.pose_keypoints[12]
            self.r_wrist = self.pose_keypoints[16]
            self.r_hip = self.pose_keypoints[24]

            self.right_arm_sX = int(self.r_shoulder['X']*640)
            self.right_arm_sY = int(self.r_shoulder['Y']*480)
            self.right_arm_eX = int(self.r_wrist['X']*640)
            self.right_arm_eY = int(self.r_wrist['Y']*480)

            self.right_arm_length = np.linalg.norm(np.array([self.right_arm_sX,self.right_arm_sY])-np.array([self.right_arm_eX,self.right_arm_eY]))

            # Pointing Metrics
            self.shoulder_distance = np.linalg.norm(np.array([self.l_shoulder['X']*640,self.l_shoulder['Y']*480])-np.array([self.r_shoulder['X']*640,self.r_shoulder['Y']*480]))
            self.l_hand_distance = np.linalg.norm(np.array([self.l_wrist['X']*640,self.l_wrist['Y']*480])-np.array([self.l_hip['X']*640,self.l_hip['Y']*480]))
            self.r_hand_distance = np.linalg.norm(np.array([self.r_wrist['X']*640,self.r_wrist['Y']*480])-np.array([self.r_hip['X']*640,self.r_hip['Y']*480]))

            self.ARM_THRESHOLD = self.shoulder_distance*.7
            self.FINGER_THRESHOLD = self.shoulder_distance/10
            self.POINTING_THRESHOLD = self.shoulder_distance/1.5


        if self.results.left_hand_landmarks:
            self.left_hand_keypoints = []
            for data_point in self.results.left_hand_landmarks.landmark:
                self.left_hand_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })

            # Drawing right arm line.
            self.l_finger_1 = self.left_hand_keypoints[5]
            self.l_finger_2 = self.left_hand_keypoints[8]

            self.left_hand_sX = int(self.l_finger_1['X']*640)
            self.left_hand_sY = int(self.l_finger_1['Y']*480)
            self.left_hand_eX = int(self.l_finger_2['X']*640)
            self.left_hand_eY = int(self.l_finger_2['Y']*480)

            self.left_finger_length = np.linalg.norm(np.array([self.left_hand_sX,self.left_hand_sY])-np.array([self.left_hand_eX,self.left_hand_eY]))

        if self.results.right_hand_landmarks:
            self.right_hand_keypoints = []
            for data_point in self.results.right_hand_landmarks.landmark:
                self.right_hand_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })

            # Drawing right arm line.
            self.r_finger_1 = self.right_hand_keypoints[5]
            self.r_finger_2 = self.right_hand_keypoints[8]

            self.right_hand_sX = int(self.r_finger_1['X']*640)
            self.right_hand_sY = int(self.r_finger_1['Y']*480)
            self.right_hand_eX = int(self.r_finger_2['X']*640)
            self.right_hand_eY = int(self.r_finger_2['Y']*480)

            self.right_finger_length = np.linalg.norm(np.array([self.right_hand_sX,self.right_hand_sY])-np.array([self.right_hand_eX,self.right_hand_eY]))


    """
    This function draws all of the pointing lines in the image.
    """
    def draw_lines(self):
        # Draw recognized pointing lines
        if self.DRAW_LINES:
            # Check if left hand is pointing
            if self.l_hand_distance >= self.POINTING_THRESHOLD:
                if self.results.pose_landmarks:
                    # Left arm
                    if self.left_arm_sX in range(0,640) and self.left_arm_sY in range(0,480) and self.left_arm_eX in range(0,640) and self.left_arm_eY in range(0,480) and self.left_arm_length > self.ARM_THRESHOLD:
                        cv2.line(self.image, (self.left_arm_sX,self.left_arm_sY), (self.left_arm_eX,self.left_arm_eY), self.LINE_COLOR, 2)
                if self.results.left_hand_landmarks:
                    # Left hand
                    if self.left_hand_sX in range(0,640) and self.left_hand_sY in range(0,480) and self.left_hand_eX in range(0,640) and self.left_hand_eY in range(0,480) and self.left_finger_length >= self.FINGER_THRESHOLD:
                        cv2.line(self.image, (self.left_hand_sX,self.left_hand_sY), (self.left_hand_eX,self.left_hand_eY), self.LINE_COLOR, 2)
            # Check if right hand is pointing
            if self.r_hand_distance >= self.POINTING_THRESHOLD:
                if self.results.pose_landmarks:
                    # Right arm
                    if self.right_arm_sX in range(0,640) and self.right_arm_sY in range(0,480) and self.right_arm_eX in range(0,640) and self.right_arm_eY in range(0,480) and self.right_arm_length > self.ARM_THRESHOLD:
                        cv2.line(self.image, (self.right_arm_sX,self.right_arm_sY), (self.right_arm_eX,self.right_arm_eY), self.LINE_COLOR, 2)
                if self.results.right_hand_landmarks:
                    # Right hand
                    if self.right_hand_sX in range(0,640) and self.right_hand_sY in range(0,480) and self.right_hand_eX in range(0,640) and self.right_hand_eY in range(0,480) and self.right_finger_length >= self.FINGER_THRESHOLD:
                        cv2.line(self.image, (self.right_hand_sX,self.right_hand_sY), (self.right_hand_eX,self.right_hand_eY), self.LINE_COLOR, 2)

        # Draw skeleton
        if self.DRAW_HAND_LINES:
            self.mp_drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if self.DRAW_POSE_LINES:
            self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)


    """
    This function determines if the provided hand or arm is pointing.
    Returns: coordinates to which the user is pointing
    """
    def determine_pointing(self):
        if self.l_hand_distance >= self.POINTING_THRESHOLD and int(self.l_wrist['X']*640) in range(0,640) and int(self.l_wrist['Y']*480) in range(0,480):
            arm = "LEFT"
        elif self.r_hand_distance >= self.POINTING_THRESHOLD and int(self.r_wrist['X']*640) in range(0,640) and int(self.r_wrist['Y']*480) in range(0,480):
            arm = "RIGHT"
        else:
            arm = "NONE"

        if arm == "LEFT":
            if self.results.left_hand_landmarks and self.left_finger_length >= self.FINGER_THRESHOLD:
                # Base the pointing direction on the finger.
                return self.get_pointing_coords(self.left_hand_sX, self.left_hand_sY, self.left_hand_eX, self.left_hand_eY)
            elif self.results.pose_landmarks and self.left_arm_length >= self.ARM_THRESHOLD:
                # Otherwise base the pointing direction on the arm.
                return self.get_pointing_coords(self.left_arm_sX, self.left_arm_sY, self.left_arm_eX, self.left_arm_eY)
            else:
                return [] # Not pointing at anything
        elif arm =="RIGHT":
            if self.results.right_hand_landmarks and self.right_finger_length >= self.FINGER_THRESHOLD:
                # Base the pointing direction on the finger.
                return self.get_pointing_coords(self.right_hand_sX, self.right_hand_sY, self.right_hand_eX, self.right_hand_eY)
            elif self.results.pose_landmarks and self.right_arm_length >= self.ARM_THRESHOLD:
                # Otherwise base the pointing direction on the arm.
                return self.get_pointing_coords(self.right_arm_sX, self.right_arm_sY, self.right_arm_eX, self.right_arm_eY)
            else:
                return [] # Not pointing at anything
        else:
            return [] # Not pointing at anything

    """
    This function determines where on the edge of the screen the line formed by the two points will cross
    returns: a list of points along the line the user is pointing - number of points determined by self.NUM_MIDPOINTS
    """
    def get_pointing_coords(self, x1, y1, x2, y2):
        # Check which direction the user is pointing
        if x1 < x2:
            pointing_right = True
        else:
            pointing_right = False

        if y1 < y2:
            pointing_down = True
        else:
            pointing_down = False

        # Find the slope of the line
        if (x2-x1) == 0.0:
            m = (y2-y1)/0.001
        else:
            m = (y2-y1)/(x2-x1)
        if m == 0.0:
            m = 0.001

        # Determine the possible edge of screen points corresponding to the line function above
        _x = 640
        _y = 480
        _x2 = 0
        _y2 = 0
        y_out = int((m*(_x-x1))+y1)
        x_out = int(((_y-y1)/m)+x1)
        y_out2 = int((m*(_x2-x1))+y1)
        x_out2 = int(((_y2-y1)/m)+x1)

        # Find the point on the edge of the screen the user is pointing to
        if pointing_right:
            if pointing_down:
                # Could be pointing right, or down
                if y_out in range(0,480):
                    x, y = _x, y_out
                    dx = x2 - _x
                else:
                    x, y = x_out, _y
                    dx = x2 - x_out
            else:
                # Could be pointing right, or up
                if y_out in range(0,480):
                    x, y = _x, y_out
                    dx = x2 - _x
                else:
                    x, y = x_out2, _y2
                    dx = x2 - x_out2
        else:
            if pointing_down:
                # Could be pointing left, or down
                if y_out2 in range(0,480):
                    x, y = _x2, y_out2
                    dx = x2 - _x2
                else:
                    x, y = x_out, _y
                    dx = x2 - x_out
            else:
                # Could be pointing left, or up
                if y_out2 in range(0,480):
                    x, y = _x2, y_out2
                    dx = x2 - _x2
                else:
                    x, y = x_out2, _y2
                    dx = x2 - x_out2

        # self.midpoints = [(x,y)]
        self.midpoints = []

        if self.DRAW_CIRCLES:
            cv2.circle(self.image, (x, y), 5, self.DOT_COLOR, 2)

        # Create other circles based on d distance
        d = dx/self.NUM_MIDPOINTS
        for i in range(self.NUM_MIDPOINTS):
            mid_x, mid_y = int(x2-(d*(i))), int((m*((x2-(d*(i)))-x1))+y1)
            mid_x_cell = mid_x//80
            mid_y_cell = mid_y//60
            self.midpoints.append((mid_x_cell,mid_y_cell))
            if self.DRAW_CIRCLES:
                cv2.circle(self.image, (mid_x, mid_y), 5, (0,255,0), 2) # Draw midpoint circles

        return self.midpoints

def main():
    rospy.init_node('pose')
    pe = PoseEstimator()
    while True:
        try:
            pe.menu()
        except KeyboardInterrupt:
            print("Shutting down...")
            cv2.destroyAllWindows()
            sys.exit()

if __name__ == "__main__":
    main()
