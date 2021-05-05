import mediapipe as mp
import cv2
import time

DRAW_LINES = True

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0) # Image size is 640*480
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Open the camera and get the frame
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform the feature detection
        results = holistic.process(image)



# Pose Keypoints
        # Extract the pose_keypoints from the results
        pose_keypoints = []
        if results.pose_landmarks:
            for data_point in results.pose_landmarks.landmark:
                pose_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })


            # Drawing left arm line.
            l_shoulder = pose_keypoints[11]
            l_wrist = pose_keypoints[15]

            print(l_shoulder['Z'])

            left_arm_sX = int(l_shoulder['X']*640)
            left_arm_sY = int(l_shoulder['Y']*480)
            left_arm_eX = int(l_wrist['X']*640)
            left_arm_eY = int(l_wrist['Y']*480)

            if left_arm_sX in range(0,640) and left_arm_sY in range(0,480) and left_arm_eX in range(0,640) and left_arm_eY in range(0,480) and DRAW_LINES:
                cv2.line(image, (left_arm_sX,left_arm_sY), (left_arm_eX,left_arm_eY), (0,0,255), 2)


            # Drawing right arm line.
            r_shoulder = pose_keypoints[12]
            r_wrist = pose_keypoints[16]

            right_arm_sX = int(r_shoulder['X']*640)
            right_arm_sY = int(r_shoulder['Y']*480)
            right_arm_eX = int(r_wrist['X']*640)
            right_arm_eY = int(r_wrist['Y']*480)

            if right_arm_sX in range(0,640) and right_arm_sY in range(0,480) and right_arm_eX in range(0,640) and right_arm_eY in range(0,480) and DRAW_LINES:
                cv2.line(image, (right_arm_sX,right_arm_sY), (right_arm_eX,right_arm_eY), (255,0,0), 2)


# Left Hand Keypoints
        left_hand_keypoints = []
        if results.left_hand_landmarks:
            for data_point in results.left_hand_landmarks.landmark:
                left_hand_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })

            # Drawing right arm line.
            l_finger_1 = left_hand_keypoints[5]
            l_finger_2 = left_hand_keypoints[8]

            left_hand_sX = int(l_finger_1['X']*640)
            left_hand_sY = int(l_finger_1['Y']*480)
            left_hand_eX = int(l_finger_2['X']*640)
            left_hand_eY = int(l_finger_2['Y']*480)

            if left_hand_sX in range(0,640) and left_hand_sY in range(0,480) and left_hand_eX in range(0,640) and left_hand_eY in range(0,480):
                cv2.line(image, (left_hand_sX,left_hand_sY), (left_hand_eX,left_hand_eY), (0,0,255), 2)


# Right Hand Keypoints
        right_hand_keypoints = []
        if results.right_hand_landmarks:
            for data_point in results.right_hand_landmarks.landmark:
                right_hand_keypoints.append({
                                     'X': data_point.x,
                                     'Y': data_point.y,
                                     'Z': data_point.z,
                                     'Visibility': data_point.visibility,
                                     })

            # Drawing right arm line.
            r_finger_1 = right_hand_keypoints[5]
            r_finger_2 = right_hand_keypoints[8]

            right_hand_sX = int(r_finger_1['X']*640)
            right_hand_sY = int(r_finger_1['Y']*480)
            right_hand_eX = int(r_finger_2['X']*640)
            right_hand_eY = int(r_finger_2['Y']*480)

            if right_hand_sX in range(0,640) and right_hand_sY in range(0,480) and right_hand_eX in range(0,640) and right_hand_eY in range(0,480):
                cv2.line(image, (right_hand_sX,right_hand_sY), (right_hand_eX,right_hand_eY), (255,0,0), 2)

        # Draw landmarks on image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


        # cv2.imshow("Webcam Feed", frame)
        cv2.imshow("Image Feed", image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
