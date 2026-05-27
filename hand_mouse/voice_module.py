import speech_recognition as sr
import pyautogui
import pyperclip
import threading

recognizer = sr.Recognizer()

# ────────────────────────────────────────────
# チューニング済みパラメータ（ここで調整）
# ────────────────────────────────────────────
recognizer.energy_threshold = 300      # 無音 / 音声の境界値（環境に合わせて調整）
recognizer.dynamic_energy_threshold = False  # 動的調整をOFF → 毎回キャリブ不要
recognizer.pause_threshold = 0.6       # 発話終了と判定するまでの無音秒数（デフォルト0.8）
recognizer.non_speaking_duration = 0.4 # 無音バッファ（デフォルト0.5）
# ────────────────────────────────────────────


def calibrate_once(duration: float = 1.0):
    """
    起動時に1回だけ環境ノイズを計測する。
    以降は dynamic_energy_threshold=False で固定値を使うため
    listen() のたびに adjust_for_ambient_noise() を呼ばなくて済む。
    """
    with sr.Microphone() as source:
        print(f"環境ノイズ計測中...（{duration}秒）")
        recognizer.adjust_for_ambient_noise(source, duration=duration)
    print(f"energy_threshold = {recognizer.energy_threshold:.1f} に設定しました\n")


def listen_voice(phrase_time_limit: int = 30) -> str | None:
    """
    音声を録音して文字列を返す。
    認識はバックグラウンドスレッドで並列実行し体感速度を上げる。
    """
    result_container: list[str | None] = [None]
    error_container:  list[str | None] = [None]

    def recognize_in_background(audio: sr.AudioData):
        """API呼び出しを別スレッドで実行"""
        try:
            result_container[0] = recognizer.recognize_google(audio, language="ja-JP")
        except sr.UnknownValueError:
            error_container[0] = "音声を聞き取れませんでした"
        except sr.RequestError as e:
            error_container[0] = f"APIエラー: {e}"

    with sr.Microphone() as source:
        print("音声待機中...")
        try:
            audio = recognizer.listen(
                source,
                timeout=3,                    # 3秒以内に話し始めなければ終了（5→3秒に短縮）
                phrase_time_limit=phrase_time_limit,
            )
        except sr.WaitTimeoutError:
            print("タイムアウト: 音声が検出されませんでした")
            return None

    # 録音完了と同時に認識スレッドを起動
    print("認識中...")
    t = threading.Thread(target=recognize_in_background, args=(audio,), daemon=True)
    t.start()
    t.join()  # 結果を待つ（UIを分離する場合はここをコールバックに変更）

    if error_container[0]:
        print(error_container[0])
        return None

    print("認識結果:", result_container[0])
    return result_container[0]


def type_text(text: str | None):
    """認識結果をアクティブウィンドウに貼り付ける"""
    if text:
        pyperclip.copy(text)
        pyautogui.hotkey("ctrl", "v")
        # 必要なら↓のコメントを外す
        # pyautogui.press("enter")


if __name__ == "__main__":
    # ── 起動時に1回だけノイズキャリブレーション ──
    calibrate_once(duration=1.0)

    # ── 連続入力ループ（Ctrl+C で終了）──
    print("Ctrl+C で終了します\n")
    try:
        while True:
            text = listen_voice(phrase_time_limit=30)
            if text:
                type_text(text)
    except KeyboardInterrupt:
        print("\n終了しました")