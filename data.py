import cv2
import numpy as np
import os
import mediapipe as mp

# 初始化Mediapipe的Holistic模型和绘图工具
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    """
    使用Mediapipe模型进行人体关键点检测
    :param image: 输入图像
    :param model: Mediapipe模型
    :return: 处理后的图像和检测结果
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_styled_landmarks(image, results):
    """
    根据检测结果绘制关键点和连接线
    :param image: 输入图像
    :param results: 检测结果
    """
    # 绘制面部关键点连接线
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # 绘制身体关键点连接线
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # 绘制左手关键点连接线
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # 绘制右手关键点连接线
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    """
    从检测结果中提取关键点坐标
    :param results: 检测结果
    :return: 关键点坐标数组
    """
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, lh, rh])


# 定义数据存储路径和动作类别
DATA_PATH = os.path.join('Video')
actions = np.array(['born', 'think', 'here', 'home', 'look', 'love', 'lie', 'why', 'even'])
no_sequences = 1
sequence_length = 900  # 30 seconds at 30 fps

# 确保保存路径存在
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# 打开摄像头并进行数据采集
cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define the codec
            video_filename = os.path.join(DATA_PATH, f"{action}_{sequence+1}.mp4")
            out = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))  # Define the VideoWriter object

            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, keypoints)

                out.write(image)  # Write the frame to the video file
                cv2.imshow('OpenCV Feed', image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            out.release()  # Release the VideoWriter object
            print(f"Finished recording {video_filename}")

cap.release()
cv2.destroyAllWindows()