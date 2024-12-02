import cv2
import mediapipe as mp
import numpy as np
from tools.landmark_handle import landmark_handle

video_num = 1
# 三个人的效果没有一个人的好，还在想办法
# also - 也
# attractive - 有吸引力的
# beautiful - 美丽的
# believe - 相信
# de - 的
# doubt - 怀疑
# dream - 梦想
# express - 表达
# eye - 眼睛
# give - 给
# handLang - (这个词可能是一个缩写或特定领域的术语，没有直接的中文翻译)
# have - 有
# many - 许多
# me - 我
# method - 方法
# no - 不
# only - 只有
# over - 结束
# please - 请
# put - 放
# say - 说
# smile - 微笑
# star - 星星
# use_accept_give - (这个词可能是一个组合词或特定上下文中的短语，没有直接的中文翻译)
# very - 非常
# watch - 观看
# you - 你

# label = ["also", "attractive", "beautiful", "believe", "de", "doubt", "dream", "express", "eye", "give", "handLang",
#          "have",
#          "many",
#          "me", "method", "no", "only", "over", "please", "put", "say", "smile", "star", "use_accept_give", "very",
#          "watch",
#          "you"]

# 想 - think
# 为什么 - why
# 出生 - born
# 这里 - here
# 家 - home
# 甚至 - even
# 父母 - parents
# 看 - look
# 羡慕 - envy
# 生活 - life
# 安全 - safety
# 和 - and
# 一样 - same
# 希望 - hope
# 爱 - love
# 接受 - accept
# 告诉 - tell
# 放弃 - give up
# 继续 - continue
# 做到 - achieve
# 知道 - know
# 难 - difficult
# 说教 - preach
# 身份 - identity
# 小心翼翼 - cautiously
# 活着 - alive
# 警惕 - vigilant
# 隐藏 - hide
# 公平 - fair
# 做错 - wrong
# 撒谎 - lie
# 糟糕 - terrible
# 惩罚 - punish
# label = ["think", "why", "born", "here", "home", "even", "parents", "look", "envy", "life", "safety", "and", "same",
#          "hope", "love", "accept", "tell", "give up", "continue", "achieve", "know", "difficult", "preach", "identity",
#          "cautiously", "alive", "vigilant", "hide", "fair", "wrong", "lie", "terrible", "punish"]
label = ["think", "here", "home", "look"]
label_num = len(label)
print("label_num:" + str(label_num))
# 模型保存地址即是label+.npz
# video_path即是label+_

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75)

for i in range(len(label)):
    print(str(i + 1) + "/" + str(len(label)) + ":" + label[i])
    data = []
    for j in range(video_num):
        # cap = cv2.VideoCapture("./Video/static/" + label[i] + "_" + str(j) + ".mp4")
        cap = cv2.VideoCapture("./Video/static/" + label[i] + "_" + "1" + ".mp4")
        ret, frame = cap.read()
        while ret is True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            results = hands.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    hand_local = []
                    for ix in range(21):
                        x = min(int(hand_landmarks.landmark[ix].x * frame.shape[1]), frame.shape[1] - 1)
                        y = min(int(hand_landmarks.landmark[ix].y * frame.shape[0]), frame.shape[0] - 1)
                        hand_local.append([x, y])
                    hand_local = landmark_handle(hand_local)
                    data.append(hand_local)

            ret, frame = cap.read()

    np.savez_compressed("./npz_files/" + label[i] + ".npz", data=data)
