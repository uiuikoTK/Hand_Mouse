import os
import sys
import cv2
import math
import time
import threading
import pyautogui
import win32gui, win32con
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from voice_module import listen_voice, type_text

# pyautogui の安全装置を無効化
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()

# ── カメラ検索 ──
def find_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

# ── ランドマーク接続情報 ──
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

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

def show_overlay(frame, text, color, w, h):
    tw = len(text) * 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (w//2 - tw//2 - 20, h//2 - 50),
                  (w//2 + tw//2 + 20, h//2 + 50), color, -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (w//2 - tw//2, h//2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)

def main():
    # ── モデルファイルのパス（exe化対応）──
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "hand_landmarker.task")

    if not os.path.exists(model_path):
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("エラー", f"hand_landmarker.task が見つかりません。\n探した場所: {model_path}")
        sys.exit(1)

    # ── カメラ選択 ──
    available_cameras = find_cameras()
    if len(available_cameras) == 0:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("エラー", "カメラが見つかりませんでした。")
        sys.exit(1)
    elif len(available_cameras) == 1:
        selected = available_cameras[0]
        print(f"カメラ {selected} を使用します。")
    else:
        import tkinter as tk
        from tkinter import simpledialog
        root = tk.Tk(); root.withdraw()
        options_str = "\n".join([f"{idx}: カメラ {idx}" for idx in available_cameras])
        while True:
            ans = simpledialog.askstring(
                "カメラ選択",
                f"使用するカメラ番号を入力してください。\n{options_str}"
            )
            if ans is None:
                sys.exit(0)
            try:
                selected = int(ans)
                if selected in available_cameras:
                    break
            except ValueError:
                pass

    # ── HandLandmarker 初期化 ──
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # ── カメラ初期化 ──
    cap = cv2.VideoCapture(selected)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── 操作設定 ──
    PINCH_THRESHOLD        = 0.06
    DIR_THRESHOLD          = 0.1
    SCROLL_AMOUNT          = 3
    CURSOR_ALPHA           = 0.3

    pinch_active           = False
    r_pinch_active         = False
    smooth_cx              = float(SCREEN_W // 2)
    smooth_cy              = float(SCREEN_H // 2)

    click_display_time     = 0
    r_click_display_time   = 0
    up_display_time        = 0
    down_display_time      = 0
    CLICK_DISPLAY_DURATION = 0.5
    DIR_DISPLAY_DURATION   = 0.3

    # ── 音声入力の状態管理 ──
    voice_state            = "idle"
    voice_state_lock       = threading.Lock()
    VOICE_HOLD_SEC         = 0.8
    peace_start_time       = 0.0
    peace_hold_active      = False

    def voice_worker():
        nonlocal voice_state
        try:
            text = listen_voice(phrase_time_limit=30)
            if text:
                type_text(text)
        finally:
            with voice_state_lock:
                voice_state = "idle"

    print("起動しました。[q]キーで終了します。")
    print(f"画面解像度: {SCREEN_W} x {SCREEN_H}")
    print("ピースサイン（人差し指＋中指を立てて0.8秒キープ）で音声入力モードに入ります。")

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

        with voice_state_lock:
            current_voice_state = voice_state

        if result.hand_landmarks:
            landmarks  = result.hand_landmarks[0]
            draw_landmarks(frame, landmarks, w, h)

            wrist      = landmarks[0]
            thumb      = landmarks[4]
            index_tip  = landmarks[8]
            index_base = landmarks[5]
            middle_tip = landmarks[12]

            thumb_ext   = finger_extended(landmarks[4],  landmarks[2],  wrist)
            index_ext   = finger_extended(landmarks[8],  landmarks[5],  wrist)
            middle_ext  = finger_extended(landmarks[12], landmarks[9],  wrist)
            ring_ext    = finger_extended(landmarks[16], landmarks[13], wrist)
            pinky_ext   = finger_extended(landmarks[20], landmarks[17], wrist)

            middle_fold = finger_folded(landmarks[12], landmarks[9],  wrist)
            ring_fold   = finger_folded(landmarks[16], landmarks[13], wrist)
            pinky_fold  = finger_folded(landmarks[20], landmarks[17], wrist)

            # スクロール姿勢（ピース誤検出防止に使う）
            scroll_pose = thumb_ext and index_ext and middle_fold and ring_fold and pinky_fold

            # ピースサイン判定（人差し指＋中指を立て、薬指・小指を折る）
            peace_pose = (
                index_ext and middle_ext
                and ring_fold and pinky_fold
                and not scroll_pose
            )

            if peace_pose and current_voice_state == "idle":
                if not peace_hold_active:
                    peace_hold_active = True
                    peace_start_time  = time.time()
                else:
                    held = time.time() - peace_start_time
                    if held >= VOICE_HOLD_SEC:
                        with voice_state_lock:
                            voice_state = "listening"
                        current_voice_state = "listening"
                        peace_hold_active   = False
                        t = threading.Thread(target=voice_worker, daemon=True)
                        t.start()
            else:
                peace_hold_active = False

            # 音声入力中は他の操作をスキップ
            if current_voice_state == "idle":

                # 全指開き → カーソル移動
                all_open = thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext \
                           and not middle_fold and not ring_fold and not pinky_fold
                if all_open:
                    is_move = True
                    # 5・9・13・17番の平均（手のひら中心）
                    palm_x = (landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 4
                    palm_y = (landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 4
                    target_x = palm_x * SCREEN_W
                    target_y = palm_y * SCREEN_H
                    smooth_cx += CURSOR_ALPHA * (target_x - smooth_cx)
                    smooth_cy += CURSOR_ALPHA * (target_y - smooth_cy)
                    cx = max(0, min(SCREEN_W - 1, int(smooth_cx)))
                    cy = max(0, min(SCREEN_H - 1, int(smooth_cy)))
                    pyautogui.moveTo(cx, cy)

                # 左クリック（親指 + 人差し指）
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

                # 右クリック（親指 + 中指）
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

                # スクロール（親指＋人差し指の2本立て）
                if scroll_pose:
                    dy = index_base.y - index_tip.y
                    if dy > DIR_THRESHOLD:
                        up_display_time = time.time()
                        pyautogui.scroll(SCROLL_AMOUNT)
                    elif dy < -DIR_THRESHOLD:
                        down_display_time = time.time()
                        pyautogui.scroll(-SCROLL_AMOUNT)

        else:
            peace_hold_active = False

        # ── オーバーレイ表示 ──
        if current_voice_state == "listening":
            show_overlay(frame, "VOICE...", (0, 180, 80), w, h)
        elif peace_hold_active:
            held  = time.time() - peace_start_time
            ratio = min(held / VOICE_HOLD_SEC, 1.0)
            bar_w = int(w * 0.6)
            bar_x = (w - bar_w) // 2
            bar_y = h - 60
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (60, 60, 60), -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * ratio), bar_y + 20), (0, 220, 80), -1)
            cv2.putText(frame, "PEACE: hold...", (bar_x, bar_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        else:
            if is_move:
                show_overlay(frame, "MOVE", (180, 0, 180), w, h)
            if time.time() - click_display_time < CLICK_DISPLAY_DURATION:
                show_overlay(frame, "CLICK!", (0, 0, 200), w, h)
            if time.time() - r_click_display_time < CLICK_DISPLAY_DURATION:
                show_overlay(frame, "RIGHT CLICK!", (0, 100, 200), w, h)
            if time.time() - up_display_time < DIR_DISPLAY_DURATION:
                show_overlay(frame, "UP", (200, 100, 0), w, h)
            if time.time() - down_display_time < DIR_DISPLAY_DURATION:
                show_overlay(frame, "DOWN", (0, 150, 100), w, h)

        cv2.putText(frame, "[q]:Quit  |  Peace sign = Voice Input", (5, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow('Hand Mouse', frame)

        # 他のウィンドウをクリックしたら最背面に移動
        try:
            hwnd = win32gui.FindWindow(None, 'Hand Mouse')
            if hwnd:
                win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0,
                    win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)
        except Exception:
            pass

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("アプリを終了しました。")
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()