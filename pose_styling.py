import mediapipe as mp
import cv2
import time

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)


        keypoints = []
        for data_point in results.pose_landmarks.landmark:
            keypoints.append({
                                 'X': data_point.x,
                                 'Y': data_point.y,
                                 'Z': data_point.z,
                                 'Visibility': data_point.visibility,
                                 })

        # print(image.shape) # 480 vertical, 640 horizontal

        # Drawing left arm line.
        l_shoulder = keypoints[11]
        l_wrist = keypoints[15]

        lsX = int(l_shoulder['X']*640)
        lsY = int(l_shoulder['Y']*480)
        leX = int(l_wrist['X']*640)
        leY = int(l_wrist['Y']*480)

        if lsX in range(0,640) and lsY in range(0,480) and leX in range(0,640) and leY in range(0,480):
            cv2.line(image, (lsX,lsY), (leX,leY), (0,0,255), 2)

        # Drawing right arm line.
        r_shoulder = keypoints[12]
        r_wrist = keypoints[16]

        rsX = int(r_shoulder['X']*640)
        rsY = int(r_shoulder['Y']*480)
        reX = int(r_wrist['X']*640)
        reY = int(r_wrist['Y']*480)

        if rsX in range(0,640) and rsY in range(0,480) and reX in range(0,640) and reY in range(0,480):
            cv2.line(image, (rsX,rsY), (reX,reY), (255,0,0), 2)


        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
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
