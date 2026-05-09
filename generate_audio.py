"""
Kokoro TTS で lessons.json の passage を読み込み、
audios/ フォルダに MP3 を一括生成するスクリプト
"""

import json
import os
import sys
import numpy as np

# ---- 設定 ----
VOICE = "af_heart"       # 音声: af_heart (女性・米英), am_adam (男性・米英) など
SPEED = 1.0              # 読み上げ速度 (0.5〜2.0)
LESSONS_JSON = "data/lessons.json"
OUTPUT_DIR   = "audios"
# --------------

def save_mp3(samples, sample_rate, path):
    """WAV 経由で MP3 に変換して保存"""
    import soundfile as sf
    import tempfile, subprocess, shutil

    # まず WAV として一時保存
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    sf.write(tmp_wav, samples, sample_rate)

    # ffmpeg があれば MP3 変換、なければ WAV のまま保存
    if shutil.which("ffmpeg"):
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_wav, "-codec:a", "libmp3lame", "-qscale:a", "2", path],
            check=True, capture_output=True
        )
        os.remove(tmp_wav)
        print(f"  → MP3 保存: {path}")
    else:
        # ffmpeg がない場合は .wav で保存（拡張子だけ変更）
        wav_path = path.replace(".mp3", ".wav")
        os.rename(tmp_wav, wav_path)
        print(f"  → WAV 保存 (ffmpeg未インストール): {wav_path}")
        print(f"     ※ lessons.json の audio_url を .wav に変更するか ffmpeg をインストールしてください")


def main():
    # lessons.json を読み込む
    with open(LESSONS_JSON, encoding="utf-8") as f:
        lessons = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Kokoro 初期化
    print("Kokoro モデルを読み込み中...")
    from kokoro_onnx import Kokoro
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    print(f"音声: {VOICE}, 速度: {SPEED}\n")

    for date_str, lesson in lessons.items():
        output_path = os.path.join(OUTPUT_DIR, f"{date_str}.mp3")

        # 既存ファイルはスキップ
        if os.path.exists(output_path) or os.path.exists(output_path.replace(".mp3", ".wav")):
            print(f"[スキップ] {date_str} (既存ファイルあり)")
            continue

        print(f"[生成中] {date_str}: {lesson['title']}")
        passage = lesson["passage"]

        # 長いテキストを段落ごとに分割して生成（メモリ節約）
        paragraphs = [p.strip() for p in passage.split("\n") if p.strip()]
        all_samples = []
        sample_rate = 24000

        for para in paragraphs:
            samples, sr = kokoro.create(para, voice=VOICE, speed=SPEED, lang="en-us")
            all_samples.append(samples)
            sample_rate = sr
            # 段落間に0.5秒の無音を挿入
            silence = np.zeros(int(sr * 0.5), dtype=samples.dtype)
            all_samples.append(silence)

        combined = np.concatenate(all_samples)
        save_mp3(combined, sample_rate, output_path)

    print("\n✅ 全ファイルの生成が完了しました！")
    print(f"   audios/ フォルダを確認してください。")


if __name__ == "__main__":
    main()
