# r2d2.py
import wave
import random
import os
from pydub import AudioSegment

def generate_r2d2_voice(text, base_dir, sample_rate=22050):
    JUNG_DECOMP = {
        'ㅐ': ['ㅏ', 'ㅣ'], 'ㅒ': ['ㅑ', 'ㅣ'], 'ㅔ': ['ㅓ', 'ㅣ'], 'ㅖ': ['ㅕ', 'ㅣ'],
        'ㅚ': ['ㅗ', 'ㅣ'], 'ㅟ': ['ㅜ', 'ㅣ'], 'ㅢ': ['ㅡ', 'ㅣ'],
        'ㅘ': ['ㅗ', 'ㅏ'], 'ㅙ': ['ㅗ', 'ㅐ'], 'ㅝ': ['ㅜ', 'ㅓ'], 'ㅞ': ['ㅜ', 'ㅔ'],
    }
    def split_jamo(char):
        if not ('가' <= char <= '힣'):
            return [char]
        base = ord(char) - ord('가')
        cho = base // (21*28)
        jung = (base % (21*28)) // 28
        jong = base % 28
        CHOS = ['ㄱ','ㄲ','ㄴ','ㄷ','ㄸ','ㄹ','ㅁ','ㅂ','ㅃ','ㅅ','ㅆ','ㅇ','ㅈ','ㅉ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        JUNGS = ['ㅏ','ㅐ','ㅑ','ㅒ','ㅓ','ㅔ','ㅕ','ㅖ','ㅗ','ㅘ','ㅙ','ㅚ','ㅛ','ㅜ','ㅝ','ㅞ','ㅟ','ㅠ','ㅡ','ㅢ','ㅣ']
        JONGS = ['','ㄱ','ㄲ','ㄳ','ㄴ','ㄵ','ㄶ','ㄷ','ㄹ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅁ','ㅂ','ㅄ','ㅅ','ㅆ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ']
        res = [CHOS[cho], JUNGS[jung]]
        if jong != 0:
            res.append(JONGS[jong])
        return res
    def flatten_jamo(jamo_seq):
        out = []
        for j in jamo_seq:
            if j in JUNG_DECOMP:
                out.extend(JUNG_DECOMP[j])
            else:
                out.append(j)
        return out

    root = os.path.join(base_dir, "sounds_korean/{0}.wav")
    vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']
    available_vowel_files = [v for v in vowels if os.path.isfile(root.format(v))]
    if not available_vowel_files:
        raise Exception("모음 wav 파일이 없습니다! sounds_korean 폴더를 확인하세요.")

    word = list(text)
    data = b""
    for w in word:
        jamo_candidates = []
        if '가' <= w <= '힣':
            jamo_candidates = flatten_jamo(split_jamo(w))
        elif w in vowels or ('ㄱ' <= w <= 'ㅎ'):
            jamo_candidates = [w]
        elif w == ' ':
            continue
        else:
            jamo_candidates = [random.choice(available_vowel_files)]
        pick = random.choice(jamo_candidates)
        try:
            with wave.open(root.format(pick), "rb") as f:
                data += f.readframes(f.getnframes())
        except Exception as e:
            continue
    audio = AudioSegment(
        data,
        sample_width=2,
        frame_rate=sample_rate,
        channels=1
    )
    return audio
