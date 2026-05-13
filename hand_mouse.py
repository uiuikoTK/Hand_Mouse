import os
import cv2
import math
import time
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── モデルファイルのパス ──
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# HandLandmarker 初期化
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ランドマーク接続情報
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # 親指
    (0,5),(5,6),(6,7),(7,8),        # 人差し指
    (0,9),(9,10),(10,11),(11,12),   # 中指
    (0,13),(13,14),(14,15),(15,16), # 薬指
    (0,17),(17,18),(18,19),(19,20), # 小指
    (5,9),(9,13),(13,17),           # 手のひら
]

# カメラ初期化
cap = cv2.VideoCapture(0)

# ピンチ判定の設定
PINCH_THRESHOLD = 0.06
pinch_active = False       # 左クリック用
r_pinch_active = False     # 右クリック用

# 表示設定
click_display_time = 0
CLICK_DISPLAY_DURATION = 0.5
r_click_display_time = 0
up_display_time = 0
down_display_time = 0
DIR_DISPLAY_DURATION = 0.3

# 人差し指の向き判定しきい値
DIR_THRESHOLD = 0.2

def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # MediaPipe で手を検出
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        draw_landmarks(frame, landmarks, w, h)

        thumb     = landmarks[4]   # 親指先端
        index_tip = landmarks[8]   # 人差し指先端
        middle_tip = landmarks[12] # 中指先端

        # ── 左クリック判定（親指4 と 人差し指先端8）──
        dist_left = math.sqrt((thumb.x - index_tip.x) ** 2 + (thumb.y - index_tip.y) ** 2)

        thumb_px  = (int(thumb.x * w), int(thumb.y * h))
        index_px  = (int(index_tip.x * w), int(index_tip.y * h))
        cv2.circle(frame, thumb_px, 10, (255, 165, 0), -1)
        cv2.circle(frame, index_px, 10, (255, 165, 0), -1)
        cv2.line(frame, thumb_px, index_px, (255, 165, 0), 2)

        if dist_left < PINCH_THRESHOLD:
            if not pinch_active:
                pinch_active = True
                click_display_time = time.time()
                # pyautogui.click() ← 左クリックしたい場合はコメントを外す
        else:
            pinch_active = False

        # ── 右クリック判定（親指4 と 中指先端12）──
        dist_right = math.sqrt((thumb.x - middle_tip.x) ** 2 + (thumb.y - middle_tip.y) ** 2)

        middle_px = (int(middle_tip.x * w), int(middle_tip.y * h))
        cv2.circle(frame, middle_px, 10, (0, 165, 255), -1)
        cv2.line(frame, thumb_px, middle_px, (0, 165, 255), 2)

        if dist_right < PINCH_THRESHOLD:
            if not r_pinch_active:
                r_pinch_active = True
                r_click_display_time = time.time()
                # pyautogui.rightClick() ← 右クリックしたい場合はコメントを外す
        else:
            r_pinch_active = False

        # ── 人差し指の向き判定（根元5 と 先端8 のY座標を比較）──
        index_base = landmarks[5]
        dy = index_base.y - index_tip.y

        if dy > DIR_THRESHOLD:
            up_display_time = time.time()
        elif dy < -DIR_THRESHOLD:
            down_display_time = time.time()

    # ── 左クリック表示 ──
    if time.time() - click_display_time < CLICK_DISPLAY_DURATION:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-120, h//2-50), (w//2+120, h//2+50), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "CLICK!", (w//2-90, h//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    # ── 右クリック表示 ──
    if time.time() - r_click_display_time < CLICK_DISPLAY_DURATION:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-160, h//2-50), (w//2+160, h//2+50), (0, 100, 200), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "RIGHT CLICK!", (w//2-150, h//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 4)

    # ── UP 表示 ──
    if time.time() - up_display_time < DIR_DISPLAY_DURATION:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-100, h//2-50), (w//2+100, h//2+50), (200, 100, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "UP", (w//2-50, h//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

    # ── DOWN 表示 ──
    if time.time() - down_display_time < DIR_DISPLAY_DURATION:
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-120, h//2-50), (w//2+120, h//2+50), (0, 150, 100), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "DOWN", (w//2-100, h//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

    cv2.putText(frame, "[q]:Quit", (5, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow('Hand Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("アプリを終了しました。")
        break

cap.release()
cv2.destroyAllWindows()
detector.close()