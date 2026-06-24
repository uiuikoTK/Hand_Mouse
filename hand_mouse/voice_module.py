import speech_recognition as sr
import pyautogui
import pyperclip
import threading

recognizer = sr.Recognizer()

recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = False
recognizer.pause_threshold = 0.6
recognizer.non_speaking_duration = 0.4



def calibrate_once(duration: float = 1.0):
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
                timeout=3,
                phrase_time_limit=phrase_time_limit,
            )
        except sr.WaitTimeoutError:
            print("タイムアウト: 音声が検出されませんでした")
            return None

    print("認識中...")
    t = threading.Thread(target=recognize_in_background, args=(audio,), daemon=True)
    t.start()
    t.join()

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


if __name__ == "__main__":
    calibrate_once(duration=1.0)

    print("Ctrl+C で終了します\n")
    try:
        while True:
            text = listen_voice(phrase_time_limit=30)
            if text:
                type_text(text)
    except KeyboardInterrupt:
        print("\n終了しました")