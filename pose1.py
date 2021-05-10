import mediapipe as mp
import cv2
import time
import numpy as np
import ast
import threading

class Pose_Estimator():
    def __init__(self):
        # Constants
        self.MODE = "MEDIUM"
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.DOT_COLOR = self.BLUE
        self.LINE_COLOR = self.BLUE
        self.DRAW_LINES = 1
        self.DRAW_POSE_LINES = 0
        self.DRAW_HAND_LINES = 0
        self.DRAW_RECTANGLES = 0
        self.ground_truth = None
        self.running = False
        self.coords = None

        # Feature detector
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic

        # Keypoint data structures
        self.pose_keypoints = []
        self.left_hand_keypoints = []
        self.right_hand_keypoints = []

    """
    This function is the main entry point to the program. This is where the user decides what to do.
    """
    def menu(self):
        choice = -1
        while True:
            print("================================")
            print("Exit program: 0")
            print("Run pose estimation: 1")
            print("Create ground truth: 2")
            print("Save ground truth to file: 3")
            print("Load ground truth from file: 4")
            try:
                choice = int(input())
            except ValueError:
                print("Error, please make a choice from the list!")
                continue
            if choice == 0:
                break
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
                continue

    """
    This function helps the user create the ground truth to use with the pose estimation.
    The user will click on points in the frame and name them in the console.
    Cease the operation by typing q with the camera feed window selected
    """
    def create_ground_truth(self):
        self.ground_truth = {}
        self.cap = cv2.VideoCapture(0)
        while self.cap.isOpened():
            self.ret, self.frame = self.cap.read()
            cv2.imshow("Camera Feed", self.frame)
            cv2.setMouseCallback("Camera Feed", self.click_event)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        print(self.ground_truth)

    """
    This is a helper function for create_ground_truth, it listens for clicks in the camera feed window
    """
    def click_event(self, event, x, y, flags, params):
        # Checking for left click
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.frame, (x, y), 5, self.DOT_COLOR, 2)
            cv2.imshow("Camera Feed", self.frame)
            if self.MODE == "SMALL":
                x = x//80
                y = y//60
            elif self.MODE == "MEDIUM":
                x = x//160
                y = y//120
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
        print("Ground truth loaded:", self.ground_truth)

    """
    This funciton checks the coords variable to see if they coorespond to anything in the ground_truth variable
    """
    def check_coords(self):
        while self.running:
            if self.coords and self.ground_truth:
                if self.coords in self.ground_truth:
                    print("You are pointing at: ", self.ground_truth[self.coords], self.coords)
                print(self.coords)
            time.sleep(1)

    """
    This function performs the pose estimation operations. It reads the camera, makes predictions, and updates internal variables based on the human pose.
    """
    def run_pose_estimation(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            self.thread = threading.Thread(target=self.check_coords)
            self.thread.start()
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                self.image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

                # Mirror the image for "selfie mode"
                # self.image = cv2.flip(self.image, 1)

                self.results = holistic.process(self.image)

                # Call the get keypoints functions
                self.get_keypoints()
                # self.get_left_hand_keypoints()
                # self.get_right_hand_keypoints()

                # Call draw line function
                # self.draw_lines()

                # Check if pointing to a certain place?
                if self.check_arms_for_pointing() == "LEFT":
                    self.coords = self.determine_pointing("LEFT")
                elif self.check_arms_for_pointing() == "RIGHT":
                    self.coords = self.determine_pointing("RIGHT")

                # Show image
                cv2.imshow("Camera Feed", self.image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    self.running = False
                    break
            self.thread.join()
            self.cap.release()
            cv2.destroyAllWindows()


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

            self.left_arm_sX = int(self.l_shoulder['X']*640)
            self.left_arm_sY = int(self.l_shoulder['Y']*480)
            self.left_arm_eX = int(self.l_wrist['X']*640)
            self.left_arm_eY = int(self.l_wrist['Y']*480)

            self.left_arm_length = np.linalg.norm(np.array([self.left_arm_sX,self.left_arm_sY])-np.array([self.left_arm_eX,self.left_arm_eY]))

            # Right Arm
            self.r_shoulder = self.pose_keypoints[12]
            self.r_wrist = self.pose_keypoints[16]

            self.right_arm_sX = int(self.r_shoulder['X']*640)
            self.right_arm_sY = int(self.r_shoulder['Y']*480)
            self.right_arm_eX = int(self.r_wrist['X']*640)
            self.right_arm_eY = int(self.r_wrist['Y']*480)

            self.right_arm_length = np.linalg.norm(np.array([self.right_arm_sX,self.right_arm_sY])-np.array([self.right_arm_eX,self.right_arm_eY]))

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
            if self.results.pose_landmarks:
                # Left arm
                if self.left_arm_sX in range(0,640) and self.left_arm_sY in range(0,480) and self.left_arm_eX in range(0,640) and self.left_arm_eY in range(0,480) and self.left_arm_length > 150:
                    cv2.line(self.image, (self.left_arm_sX,self.left_arm_sY), (self.left_arm_eX,self.left_arm_eY), self.LINE_COLOR, 2)
                # Right arm
                if self.right_arm_sX in range(0,640) and self.right_arm_sY in range(0,480) and self.right_arm_eX in range(0,640) and self.right_arm_eY in range(0,480) and self.right_arm_length > 150:
                    cv2.line(self.image, (self.right_arm_sX,self.right_arm_sY), (self.right_arm_eX,self.right_arm_eY), self.LINE_COLOR, 2)

            if self.results.left_hand_landmarks:
                # Left hand
                if self.left_hand_sX in range(0,640) and self.left_hand_sY in range(0,480) and self.left_hand_eX in range(0,640) and self.left_hand_eY in range(0,480) and self.left_finger_length >= 40:
                    cv2.line(self.image, (self.left_hand_sX,self.left_hand_sY), (self.left_hand_eX,self.left_hand_eY), self.LINE_COLOR, 2)

            if self.results.right_hand_landmarks:
                # Right hand
                if self.right_hand_sX in range(0,640) and self.right_hand_sY in range(0,480) and self.right_hand_eX in range(0,640) and self.right_hand_eY in range(0,480) and self.right_finger_length >= 40:
                    cv2.line(self.image, (self.right_hand_sX,self.right_hand_sY), (self.right_hand_eX,self.right_hand_eY), self.LINE_COLOR, 2)

        # Draw skeleton
        self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        if self.DRAW_HAND_LINES:
            self.mp_drawing.draw_landmarks(self.image, self.results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(self.image, self.results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
        if self.DRAW_POSE_LINES:
            self.mp_drawing.draw_landmarks(self.image, self.results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

    """
    This function determines which arm should checked for pointing, if any
    returns: string: LEFT or RIGHT or NONE
    """
    def check_arms_for_pointing(self):
        return "LEFT"


    """
    This function determines if the left hand or arm is pointing.
    Returns: coordinates to which the user is pointing
    """
    def determine_pointing(self, arm):
        if arm == "LEFT":
            if self.results.left_hand_landmarks and self.left_finger_length >= 40:
                # Base the pointing direction on the finger.
                return self.is_pointing(self.left_hand_sX, self.left_hand_sY, self.left_hand_eX, self.left_hand_eY)
            elif self.results.pose_landmarks and self.left_arm_length >= 150:
                # Otherwise base the pointing direction on the arm.
                return self.is_pointing(self.left_arm_sX, self.left_arm_sY, self.left_arm_eX, self.left_arm_eY)
            else:
                return "Not pointing"
        else:
            return "right arm is pointing"

    """
    This function determines where on the edge of the screen the line formed by the two points will cross
    returns: coordinate pair that determines where the user is pointing
    """
    def is_pointing(self, x1, y1, x2, y2):
        if x1 < x2:
            pointing_right = True
        else:
            pointing_right = False

        if y1 < y2:
            pointing_down = True
        else:
            pointing_down = False


        if (x2-x1) == 0.0:
            m = (y2-y1)/0.001
        else:
            m = (y2-y1)/(x2-x1)
        if m == 0.0:
            m = 0.001

        _x = 640
        _y = 480
        _x2 = 0
        _y2 = 0
        y_out = int((m*(_x-x1))+y1)
        x_out = int(((_y-y1)/m)+x1)
        y_out2 = int((m*(_x2-x1))+y1)
        x_out2 = int(((_y2-y1)/m)+x1)

        if pointing_right:
            if pointing_down:
                # Could be pointing right, or down
                if y_out in range(0,480):
                    cv2.circle(self.image, (_x, y_out), 5, self.DOT_COLOR, 2) # RIGHT
                    pointing = "RIGHT"
                else:
                    cv2.circle(self.image, (x_out, _y), 5, self.DOT_COLOR, 2) # DOWN
                    pointing = "DOWN"
            else:
                # Could be pointing right, or up
                if y_out in range(0,480):
                    cv2.circle(self.image, (_x, y_out), 5, self.DOT_COLOR, 2) # RIGHT
                    pointing = "RIGHT"
                else:
                    cv2.circle(self.image, (x_out2, _y2), 5, self.DOT_COLOR, 2) # UP
                    pointing = "UP"
        else:
            if pointing_down:
                # Could be pointing left, or down
                if y_out2 in range(0,480):
                    cv2.circle(self.image, (_x2, y_out2), 5, self.DOT_COLOR, 2) # LEFT
                    pointing = "LEFT"
                else:
                    cv2.circle(self.image, (x_out, _y), 5, self.DOT_COLOR, 2) # DOWN
                    pointing = "DOWN"
            else:
                # Could be pointing left, or up
                if y_out2 in range(0,480):
                    cv2.circle(self.image, (_x2, y_out2), 5, self.DOT_COLOR, 2) # LEFT
                    pointing = "LEFT"
                else:
                    cv2.circle(self.image, (x_out2, _y2), 5, self.DOT_COLOR, 2) # UP
                    pointing = "UP"

        if self.MODE == "SMALL":
            if pointing == "LEFT":
                x_pos = _x2//80
                y_pos = y_out2//60
                p1 = (x_pos*80,y_pos*60)
                p2 = ((x_pos+1)*80, (y_pos+1)*60)
            elif pointing == "RIGHT":
                x_pos = _x//80
                x_pos -= 1
                y_pos = y_out//60
                p1 = ((x_pos+1)*80,y_pos*60)
                p2 = (x_pos*80, (y_pos+1)*60)
            elif pointing == "UP":
                x_pos = x_out2//80
                y_pos = _y2//60
                p1 = (x_pos*80,y_pos*60)
                p2 = ((x_pos+1)*80, (y_pos+1)*60)
            elif pointing == "DOWN":
                x_pos = x_out//80
                y_pos = _y//60
                y_pos -= 1
                p1 = (x_pos*80,(y_pos+1)*60)
                p2 = ((x_pos+1)*80, y_pos*60)
        elif self.MODE == "MEDIUM":
            if pointing == "LEFT":
                x_pos = _x2//160
                y_pos = y_out2//120
                p1 = (x_pos*160,y_pos*120)
                p2 = ((x_pos+1)*160, (y_pos+1)*120)
            elif pointing == "RIGHT":
                x_pos = _x//160
                x_pos -= 1
                y_pos = y_out//120
                p1 = ((x_pos+1)*160,y_pos*120)
                p2 = (x_pos*160, (y_pos+1)*120)
            elif pointing == "UP":
                x_pos = x_out2//160
                y_pos = _y2//120
                p1 = (x_pos*160,y_pos*120)
                p2 = ((x_pos+1)*160, (y_pos+1)*120)
            elif pointing == "DOWN":
                x_pos = x_out//160
                y_pos = _y//120
                y_pos -= 1
                p1 = (x_pos*160,(y_pos+1)*120)
                p2 = ((x_pos+1)*160, y_pos*120)
        if self.DRAW_RECTANGLES:
            cv2.rectangle(self.image, p1, p2, self.RED, 2)
        return (x_pos,y_pos)

def main():
    est = Pose_Estimator()
    est.menu()

if __name__ == "__main__":
    main()
