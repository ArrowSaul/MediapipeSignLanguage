import cv2
import mediapipe as mp
import time
import torch as t
from model import HandModel
from tools.landmark_handle import landmark_handle
from tools.draw_landmarks import draw_landmarks
from tools.calc_landmark_list import calc_landmark_list
from tools.draw_bounding_rect import draw_bounding_rect
import numpy as np
from tools.draw_rect_text import draw_rect_txt

# 加载模型的路径
model_path = 'checkpoints/model_test1.pth'
# model_path = 'checkpoints/model_39.pth'
# 标签列表，用于最后的预测结果映射
label = ["also", "attractive", "beautiful", "believe", "de", "doubt", "dream", "express", "eye", "give", "handLang",
         "have",
         "many",
         "me", "method", "no", "only", "over", "please", "put", "say", "smile", "star", "use_accept_give", "very",
         "watch",
         "you"]
# label = ["think", "why", "here", "home", "even", "look", "life", "and", "same",
#          "hope", "love",
#          "accept", "tell",
#          "give_up", "continue", "achieve",
#          "vigilant", "lie"]
label_num = len(label)

# 背景标志，决定是否使用背景
background_flag = 0
background_color = 128

# 初始化手部模型
model = HandModel()
state_dict = t.load(model_path)
model.load_state_dict(state_dict)

# 使用MediaPipe的手部解决方案
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)
# 打开摄像头
cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./Video/static/eye_1.mp4")

# 初始化帧计数器
FrameCount = 0
# 记录开始时间
time1 = time.time()
# 初始化帧率
fps = 0

# 无限循环，用于处理视频流中的每一帧
while True:
    # 读取视频流的一帧
    ret, frame = cap.read()
    # 将帧从BGR颜色空间转换到RGB颜色空间
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # 水平翻转帧，以适应镜像效果
    frame = cv2.flip(frame, 1)
    # 使用Hands解决方案处理帧，检测手部 landmarks
    results = hands.process(frame)
    # 将帧从RGB颜色空间转换回BGR颜色空间，以便OpenCV显示
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 初始化手部关键点列表
    hand_local = []
    # 如果检测到手部关键点
    if results.multi_hand_landmarks:
        # 遍历每一组手部关键点
        for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # 遍历每个关键点，获取其坐标
            for i in range(21):
                x = min(int(hand_landmarks.landmark[i].x * frame.shape[1]), frame.shape[1] - 1)
                y = min(int(hand_landmarks.landmark[i].y * frame.shape[0]), frame.shape[0] - 1)
                hand_local.append([x, y])

            # 如果设置了背景标志，则创建一个灰色背景
            if background_flag:
                frame = np.zeros(frame.shape, np.uint8)
                frame.fill(128)

            # 自己填骨架的色
            draw_landmarks(frame, hand_local)
            # 绘制手部关键点的边界框
            brect = draw_bounding_rect(frame, hand_local)
            # brect是框架的四个点坐标
            # 对手部关键点进行处理
            hand_local = landmark_handle(hand_local)

    # # 如果手部关键点列表不为空
    # if hand_local:
    #     # 使用模型预测手部姿势
    #     output = model(t.tensor(hand_local))
    #     # 获取预测结果中概率最大的类别和概率值
    #     index, value = output.topk(1)[1][0][0], output.topk(1)[0][0][0]
    #     this_label = label[index]
    #     # 在帧上绘制预测结果
    #     draw_rect_txt(frame, this_label + ":" + str(value), brect)
    #
    #     # 如果概率值大于9
    #     if value > 9:
    #         # 在帧上绘制类别标签
    #         cv2.putText(frame,
    #                     this_label,
    #                     (30, 50),
    #                     cv2.FONT_HERSHEY_SIMPLEX,
    #                     1.5,
    #                     (255, 255, 255),
    #                     3)
    # 如果手部关键点列表不为空
    if hand_local:
        # 使用模型预测手部姿势
        output = model(t.tensor(hand_local).float())  # 确保输入数据是float类型
        # 获取预测结果中概率最大的类别和概率值
        index = output.topk(1)[1][0].item()  # 获取最大值的索引，并转换为Python数字
        value = output.topk(1)[0][0].item()  # 获取最大值，并转换为Python数字
        this_label = label[index]
        # 在帧上绘制预测结果
        draw_rect_txt(frame, this_label + ":" + str(value), brect)

        # 如果概率值大于9
        if value > 9:
            # 在帧上绘制类别标签
            cv2.putText(frame,
                        this_label,
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 255, 255),
                        3)

    # 记录当前时间
    time2 = time.time()
    # 增加帧计数
    FrameCount += 1
    # 每0.5秒更新一次帧率
    if time2 - time1 >= 0.5:
        if FrameCount > 0:
            # 计算帧率
            fps = round(FrameCount / (time2 - time1), 2)
            # 重置开始时间
            time1 = time.time()
            # 重置帧计数器
            FrameCount = 0

    # 在帧上绘制帧率
    cv2.putText(frame,
                str(fps),
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1)

    # 显示处理后的帧
    cv2.imshow('MediaPipe Hands', frame)
    # 如果按下ESC键，退出循环
    if cv2.waitKey(1) & 0xFF == 18:
        break

# 释放视频捕获对象
cap.release()

