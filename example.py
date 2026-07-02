"""
En basit kullanım: bir WAV dosyası + o WAV'da söylenen metni ver, viseme
zaman çizelgesini al.

Kullanım:
    python example.py ornek.wav "Merhaba, nasılsın?"
"""
import sys
import json

import soundfile as sf

from vislign import align_visemes, finalize


def main():
    if len(sys.argv) != 3:
        print("Kullanım: python example.py <ses.wav> \"<metin>\"")
        sys.exit(1)

    wav_path, text = sys.argv[1], sys.argv[2]

    data, sr = sf.read(wav_path, dtype="int16", always_2d=True)
    pcm = data[:, 0].tobytes()  # mono, 16-bit PCM

    raw_cues = align_visemes(pcm, sr, text)
    cues = finalize(raw_cues)

    print(f"{len(cues)} viseme cue, {cues[-1]['end'] if cues else 0:.2f} saniye")
    print(json.dumps(cues[:10], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
