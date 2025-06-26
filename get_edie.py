# edie.py
import os
import random
from typing import Optional

from pydub import AudioSegment

# ------------------------------------------------------------------------
# 환경 경로 설정
# ------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
SOUND_ROOT = os.path.join(BASE_DIR, "new_emotion_sounds")  # ← 필요하면 변경
DEFAULT_EMOTION = "neutral"                                # 하위 폴더 이름
DEFAULT_RATE = 44_100                                      # Hz

def _scan_emotion_folder(emotion: str = DEFAULT_EMOTION):
    """
    (1) '_'(언더바) WAV ↔ 공백, (2) 나머지 WAV 파일 리스트를 반환
        - 폴더 구조: new_emotion_sounds/{emotion}/*.wav
    """
    if emotion == "negative":
        sub = random.choice(["strong", "weak"])   # 필요하면 "week" 로 변경
        emotion_path = os.path.join(emotion, sub) # 예: negative/strong
    else:
        emotion_path = emotion    
        
    target_dir = os.path.join(SOUND_ROOT, emotion_path)
    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"폴더 없음: {target_dir}")

    underscore_wav = os.path.join(target_dir, "_.wav")
    if not os.path.isfile(underscore_wav):
        raise FileNotFoundError("'_.wav' 가 없습니다 → 공백용 음원 필요")

    normal_wavs = [
        os.path.join(target_dir, f)
        for f in os.listdir(target_dir)
        if f.endswith(".wav") and f != "_.wav"
    ]
    if not normal_wavs:
        raise FileNotFoundError("'_.wav' 를 제외한 .wav 파일이 없습니다")

    return underscore_wav, normal_wavs


def generate_edie_voice(
    text: str,
    emotion: str = DEFAULT_EMOTION,
    random_seed: Optional[int] = None,
    normalize_rate: int = DEFAULT_RATE,
) -> Optional[AudioSegment]:
    """
    ➡ 텍스트를 EDIE sound 로 합성해 AudioSegment 로 반환
       - 같은 문장 안에서는 같은 글자 ↔ 같은 음원(고정 랜덤)
       - `emotion` 폴더 안의 .wav 들을 사용
    """
    if not text.strip():
        return None

    # 고정 랜덤 시드(선택)
    if random_seed is not None:
        random.seed(random_seed)

    underscore_wav, normal_wavs = _scan_emotion_folder(emotion)

    char2wav = {}           # 고정 매핑
    result = None           # 최종 AudioSegment

    for ch in text:
        if ch == " ":
            wav_path = underscore_wav
        else:
            if ch not in char2wav:
                char2wav[ch] = random.choice(normal_wavs)
            wav_path = char2wav[ch]

        try:
            seg = AudioSegment.from_wav(wav_path)
            seg = seg.set_frame_rate(normalize_rate)
            result = seg if result is None else result + seg
        except Exception as e:
            print(f"[경고] {wav_path} 로드 실패 → {e}")

    return result
