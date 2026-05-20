import os
import cv2
import math
import time
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# pyautogui の安全装置を無効化（画面端でも止まらない）
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── スクリーンサイズ取得 ──
SCREEN_W, SCREEN_H = pyautogui.size()

# ── 利用可能なカメラを検索 ──
def find_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

available_cameras = find_cameras()

if len(available_cameras) == 0:
    print("カメラが見つかりませんでした。")
    exit()
elif len(available_cameras) == 1:
    selected = available_cameras[0]
    print(f"カメラ {selected} を使用します。")
else:
    print("利用可能なカメラ:")
    for idx in available_cameras:
        print(f"  [{idx}] カメラ {idx}")
    while True:
        try:
            selected = int(input(f"使用するカメラ番号を入力してください ({available_cameras[0]}～{available_cameras[-1]}): "))
            if selected in available_cameras:
                break
            else:
                print("リストにない番号です。もう一度入力してください。")
        except ValueError:
            print("数字を入力してください。")

# ── モデルファイルのパス ──
# 事前に以下からダウンロードしてこのスクリプトと同じフォルダに置いてください
# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ── HandLandmarker 初期化 ──
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = mp_vision.HandLandmarker.create_from_options(options)

# ── ランドマーク接続情報 ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── カメラ初期化 ──
cap = cv2.VideoCapture(selected)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── ピンチ・操作設定 ──
PINCH_THRESHOLD    = 0.06
DIR_THRESHOLD      = 0.1
SCROLL_AMOUNT      = 3        # 1回のスクロール量
CURSOR_ALPHA       = 0.3      # EMA平滑化係数（0〜1、小さいほど滑らか・遅延大）

pinch_active       = False
r_pinch_active     = False
smooth_cx = float(SCREEN_W // 2)  # EMA平滑化後のX（float保持）
smooth_cy = float(SCREEN_H // 2)  # EMA平滑化後のY

# ── 表示設定 ──
click_display_time   = 0
r_click_display_time = 0
up_display_time      = 0
down_display_time    = 0
move_active          = False
CLICK_DISPLAY_DURATION = 0.5
DIR_DISPLAY_DURATION   = 0.3

def dist2d(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def draw_landmarks(frame, landmarks, w, h):
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 200, 0), 2)
    for pt in points:
        cv2.circle(frame, pt, 4, (255, 255, 255), -1)

def finger_extended(tip, base, wrist):
    return dist2d(tip, wrist) > dist2d(base, wrist)

def finger_folded(tip, base, wrist):
    return dist2d(tip, wrist) < dist2d(base, wrist) * 1.6

print("起動しました。[q]キーで終了します。")
print(f"画面解像度: {SCREEN_W} x {SCREEN_H}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    is_directing = (
        time.time() - up_display_time   < DIR_DISPLAY_DURATION or
        time.time() - down_display_time < DIR_DISPLAY_DURATION
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = detector.detect(mp_image)

    is_move = False

    if result.hand_landmarks:
        landmarks  = result.hand_landmarks[0]
        draw_landmarks(frame, landmarks, w, h)

        wrist      = landmarks[0]
        thumb      = landmarks[4]
        index_tip  = landmarks[8]
        index_base = landmarks[5]
        middle_tip = landmarks[12]

        # ── 各指の伸び・折れ判定 ──
        thumb_ext  = finger_extended(landmarks[4],  landmarks[2],  wrist)
        index_ext  = finger_extended(landmarks[8],  landmarks[5],  wrist)
        middle_ext = finger_extended(landmarks[12], landmarks[9],  wrist)
        ring_ext   = finger_extended(landmarks[16], landmarks[13], wrist)
        pinky_ext  = finger_extended(landmarks[20], landmarks[17], wrist)

        middle_fold = finger_folded(landmarks[12], landmarks[9],  wrist)
        ring_fold   = finger_folded(landmarks[16], landmarks[13], wrist)
        pinky_fold  = finger_folded(landmarks[20], landmarks[17], wrist)

        # ── 全指開き → カーソル移動 ──
        # 中指・薬指・小指が少しでも曲がったらMOVE終了してカーソルを固定
        all_open = thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext \
                   and not middle_fold and not ring_fold and not pinky_fold
        if all_open:
            is_move = True
            # 人差し指先端の位置をスクリーン座標にマッピング
            target_x = index_tip.x * SCREEN_W
            target_y = index_tip.y * SCREEN_H
            # EMA（指数移動平均）平滑化：毎フレーム少しずつ目標に近づく
            smooth_cx += CURSOR_ALPHA * (target_x - smooth_cx)
            smooth_cy += CURSOR_ALPHA * (target_y - smooth_cy)
            cx = max(0, min(SCREEN_W - 1, int(smooth_cx)))
            cy = max(0, min(SCREEN_H - 1, int(smooth_cy)))
            pyautogui.moveTo(cx, cy)

        # ── 左クリック（親指 + 人差し指ピンチ）──
        thumb_px = (int(thumb.x * w), int(thumb.y * h))
        index_px = (int(index_tip.x * w), int(index_tip.y * h))
        cv2.circle(frame, thumb_px, 10, (255, 165, 0), -1)
        cv2.circle(frame, index_px, 10, (255, 165, 0), -1)
        cv2.line(frame, thumb_px, index_px, (255, 165, 0), 2)

        dist_left = dist2d(thumb, index_tip)
        if dist_left < PINCH_THRESHOLD and not is_directing:
            if not pinch_active:
                pinch_active = True
                click_display_time = time.time()
                pyautogui.click()
        else:
            pinch_active = False

        # ── 右クリック（親指 + 中指ピンチ）──
        middle_px = (int(middle_tip.x * w), int(middle_tip.y * h))
        cv2.circle(frame, middle_px, 10, (0, 165, 255), -1)
        cv2.line(frame, thumb_px, middle_px, (0, 165, 255), 2)

        dist_right = dist2d(thumb, middle_tip)
        if dist_right < PINCH_THRESHOLD and not is_directing:
            if not r_pinch_active:
                r_pinch_active = True
                r_click_display_time = time.time()
                pyautogui.rightClick()
        else:
            r_pinch_active = False

        # ── 人差し指＋親指の2本が立っている → スクロール方向判定 ──
        # （中指・薬指・小指は折れていること）
        scroll_pose = thumb_ext and index_ext and middle_fold and ring_fold and pinky_fold
        if scroll_pose:
            dy = index_base.y - index_tip.y
            if dy > DIR_THRESHOLD:
                up_display_time = time.time()
                pyautogui.scroll(SCROLL_AMOUNT)
            elif dy < -DIR_THRESHOLD:
                down_display_time = time.time()
                pyautogui.scroll(-SCROLL_AMOUNT)

    # ── オーバーレイ表示 ──
    def show_overlay(text, color):
        tw = len(text) * 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2 - tw//2 - 20, h//2 - 50), (w//2 + tw//2 + 20, h//2 + 50), color, -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, text, (w//2 - tw//2, h//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

    if is_move:
        show_overlay("MOVE", (180, 0, 180))
    if time.time() - click_display_time < CLICK_DISPLAY_DURATION:
        show_overlay("CLICK!", (0, 0, 200))
    if time.time() - r_click_display_time < CLICK_DISPLAY_DURATION:
        show_overlay("RIGHT CLICK!", (0, 100, 200))
    if time.time() - up_display_time < DIR_DISPLAY_DURATION:
        show_overlay("UP", (200, 100, 0))
    if time.time() - down_display_time < DIR_DISPLAY_DURATION:
        show_overlay("DOWN", (0, 150, 100))

    cv2.putText(frame, "[q]:Quit", (5, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow('Hand Mouse', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("アプリを終了しました。")
        break

cap.release()
cv2.destroyAllWindows()
detector.close()