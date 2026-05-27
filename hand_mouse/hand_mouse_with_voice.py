import speech_recognition as sr
import pyautogui
import pyperclip

# 音声認識オブジェクト
recognizer = sr.Recognizer()

def listen_voice(phrase_time_limit=30):  # 最大30秒録音

    with sr.Microphone() as source:

        print("音声待機中...（最大30秒）")

        # 周囲の雑音調整
        recognizer.adjust_for_ambient_noise(source)

        # 音声取得（無音になるまで、最大30秒）
        audio = recognizer.listen(
            source,
            timeout=5,                        # 5秒以内に話し始めなければ終了
            phrase_time_limit=phrase_time_limit
        )

    try:
        # 日本語音声認識
        text = recognizer.recognize_google(
            audio,
            language='ja-JP'
        )

        print("認識結果:", text)

        return text

    except sr.UnknownValueError:
        print("音声を聞き取れませんでした")
        return None

    except sr.RequestError as e:
        print(f"APIエラー: {e}")
        return None


# 文字入力する関数
def type_text(text):

    if text:
        pyperclip.copy(text)
        pyautogui.hotkey('ctrl', 'v')

        # 必要ならEnter
        # pyautogui.press("enter")

if __name__ == "__main__":

    text = listen_voice(phrase_time_limit=30)  # 秒数はここで変更可能

    if text:
        type_text(text)