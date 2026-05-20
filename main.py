import subprocess
import sys
import os

# このファイルと同じフォルダのhand_mouse.pyを実行する
script_dir = os.path.dirname(os.path.abspath(__file__))
hand_mouse_path = os.path.join(script_dir, "hand_mouse.py")

if not os.path.exists(hand_mouse_path):
    print("エラー: hand_mouse.py が見つかりません。")
    print(f"  探した場所: {hand_mouse_path}")
    sys.exit(1)

print("Hand Mouse を起動します...")
print("終了するにはカメラウィンドウで [q] を押してください。")
print()

subprocess.run([sys.executable, hand_mouse_path])
