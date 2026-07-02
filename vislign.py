"""
Vislign — text + audio -> portable visemes via FORCED ALIGNMENT (torchaudio MMS_FA).

Acoustic lip-sync tools (Rhubarb and friends) guess mouth shapes from the audio
waveform alone — they never see the text, so they're blind and limited to a
handful of shapes. Vislign already knows the text (you fed it to your TTS), so
it force-aligns text<->audio and reads each letter's real timing straight out of
the alignment model. That gives much more accurate, much richer lip-sync — and
because it's just text+audio, it works with ANY TTS engine, not a specific one.

This build ships a Turkish letter->viseme table and Turkish-specific text
normalization (numbers, abbreviations, %, dates/times, İ/I casing, ğ handling).
Porting to another language means swapping the `_VIS` table and `normalize_tr`.

Two stages:
  align_visemes(pcm, sr, text) -> raw cues [{offset,end,value,id,ph}]  (true phoneme timing)
  finalize(cues, openness, base_mix) -> FINISHED animation cues, ready to play:
       each cue fully specifies {start,end,viseme,rigId,alpha,mix,phoneme} with
       anticipation lead + pause smoothing already baked in. `viseme` is a
       portable shape name any rig can map; `rigId` is this build's reference
       rig id (remap it to your own rig, or ignore it and just use `viseme`).
"""
import re
import threading
import numpy as np
import torch

_SR = 16000  # MMS_FA works at 16 kHz

# Playback feel, baked into the finished cues so screen == downloaded file.
LEAD = 0.04        # mouth leads the sound ~40 ms (humans anticipate); not after a rest
PAUSE_MIX = 0.15   # softer blend closing into / opening out of a pause (sentence ends)
OPEN_MAP = {2: 0.72}  # per-shape openness scale (a/AA opens a touch less)

# Turkish letter -> ASCII for the aligner dict (1 char -> 1 char, indices preserved
# so we can still read the ORIGINAL letter's viseme back after alignment).
_ROMANIZE = str.maketrans({
    "ç": "c", "ğ": "g", "ı": "i", "ö": "o", "ş": "s", "ü": "u",
    "â": "a", "î": "i", "û": "u", "Â": "a", "Î": "i", "Û": "u",
})

# Turkish letter -> (portable viseme name, reference-rig viseme id). Reference
# rig calibration: 0=rest 2=a 4=e 6=i/ı(wide) 7=u/ü(round-narrow) 8=o/ö(round-open)
# 21=closed(MBP) 18=F/V 19=T/D/N/R 15=S/Z 16=Ş/Ç/C/J 20=K/G 14=L 12=H
_VIS = {
    "a": ("AA", 2), "e": ("E", 4),
    "ı": ("II", 6), "i": ("II", 6), "y": ("II", 6),
    "o": ("O", 8), "ö": ("O", 8),
    "u": ("U", 7), "ü": ("U", 7),
    "p": ("MBP", 21), "b": ("MBP", 21), "m": ("MBP", 21),
    "f": ("FV", 18), "v": ("FV", 18),
    "t": ("TD", 19), "d": ("TD", 19), "n": ("TD", 19), "r": ("TD", 19),
    "s": ("SZ", 15), "z": ("SZ", 15),
    "ş": ("CH", 16), "ç": ("CH", 16), "c": ("CH", 16), "j": ("CH", 16),
    "k": ("KG", 20), "g": ("KG", 20), "ğ": ("KG", 20),
    "h": ("H", 12), "l": ("L", 14),
    # şapkalı ünlüler (kâğıt, îma) + yabancı harfler (web, taxi): rest'e düşmesinler
    "â": ("AA", 2), "î": ("II", 6), "û": ("U", 7),
    "w": ("FV", 18), "x": ("SZ", 15), "q": ("KG", 20),
}

_VOWELS = set("aeıioöuüâîû")


def _word_visemes(letters):
    """Harf listesi -> (viseme adı, rig id) listesi, Türkçe bağlamla:
    'ğ' (yumuşak g) çoğunlukla sessizdir ve ÖNCEKİ ÜNLÜYÜ UZATIR — gerçek ağız
    ünlü şeklinde kalır ('Kimliğini'≈'kimliini', 'dağa'≈'daa'). Bu yüzden ğ,
    önceki ünlünün viseme'ini alır (KG'ye gitmez). Kelime başında ğ olmaz."""
    out, last_v = [], None
    for ch in letters:
        if ch == "ğ" and last_v:
            out.append(_VIS[last_v])
        else:
            out.append(_VIS.get(ch, ("rest", 0)))
        if ch in _VOWELS:
            last_v = ch
    return out


# --- Türkçe metin normalizasyonu (yalnız HİZALAYICI girdisi; TTS metni değişmez) --
_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]
_TR_LOWER = str.maketrans({"İ": "i", "I": "ı"})  # Python .lower() Türkçe İ/I'yı bilmez


def _num_tr(n: int) -> str:
    """0..999,999,999 -> Türkçe okunuş (2024 -> iki bin yirmi dört)."""
    if n == 0:
        return "sıfır"
    if n >= 1_000_000:
        m, r = divmod(n, 1_000_000)
        return (_num_tr(m) + " milyon" + ((" " + _num_tr(r)) if r else ""))
    parts = []
    if n >= 1000:
        b, n = divmod(n, 1000)
        parts.append("bin" if b == 1 else _num_tr(b) + " bin")
    if n >= 100:
        y, n = divmod(n, 100)
        parts.append("yüz" if y == 1 else _ONES[y] + " yüz")
    if n >= 10:
        t, n = divmod(n, 10)
        parts.append(_TENS[t])
    if n:
        parts.append(_ONES[n])
    return " ".join(parts)


def _spell_digits(s: str) -> str:
    """Rakam rakam okunuş (telefon numarası / baş-sıfırlı diziler)."""
    return " ".join(_num_tr(int(d)) for d in s)


# Türk alfabesi harf adları (kısaltma hecelenmesi: TC->"te ce", KVKK->"ka ve ka ka").
_LETTER_NAME = {"a": "a", "b": "be", "c": "ce", "ç": "çe", "d": "de", "e": "e", "f": "fe",
                "g": "ge", "ğ": "yumuşak ge", "h": "he", "ı": "ı", "i": "i", "j": "je",
                "k": "ka", "l": "le", "m": "me", "n": "ne", "o": "o", "ö": "ö", "p": "pe",
                "r": "re", "s": "se", "ş": "şe", "t": "te", "u": "u", "ü": "ü", "v": "ve",
                "y": "ye", "z": "ze", "w": "çift ve", "x": "iks", "q": "kü"}
_DOT_ABBREV = {"dr": "doktor", "prof": "profesör", "doç": "doçent", "av": "avukat"}


def _abbrev_pass(text: str) -> str:
    """TAMAMEN-büyük kısaltmaları harf adlarıyla aç: TTS bunları heceler ("te-ce"),
    harf-harf hizalama ise ünlüsüz kalır -> ünlü ağzı hiç açılmaz (ölçüldü: KVKK).
    Kural: ünlüsüz kısaltma (SMS, PDF, KVKK) her boyda; ünlülü ise yalnız ≤3 harf
    (TC, ABD) — İBAN gibi kelime-gibi okunanlar dokunulmaz."""
    def repl(m):
        low = m.group(0).translate(_TR_LOWER).lower()
        if (not any(ch in _VOWELS for ch in low)) or len(low) <= 3:
            return " ".join(_LETTER_NAME.get(ch, ch) for ch in low)
        return m.group(0)
    return re.sub(r"\b[A-ZÇĞİÖŞÜ]{2,6}\b", repl, text)


def normalize_tr(text: str) -> str:
    """Hizalama girdisi için Türkçe normalizasyon. MMS_FA sözlüğünde rakam YOK —
    rakamlar sözcüğe çevrilmezse o bölge hizalanamaz ve ağız donar (ölçüldü:
    '15 Mayıs 2024...' cümlesinde 1.6s sesli-ama-kapalı). Kurallar:
    kısaltma hecele, TR küçük-harf (İ→i, I→ı), Dr.→doktor, para sembolü, binlik
    nokta at, % (ondalıklı dahil)→'yüzde', ondalık ','→'virgül', tarih/saat
    ayracı ( . / : ) boşluğa, ≤4 hane & baş-sıfırsız → sayı okunuşu,
    uzun/baş-sıfırlı → rakam rakam."""
    t = _abbrev_pass(text)
    t = t.translate(_TR_LOWER).lower()
    t = re.sub(r"\b(dr|prof|doç|av)\.", lambda m: _DOT_ABBREV[m.group(1)] + " ", t)
    # para: ₺150 / 150₺ / 150 TL -> "... lira" (sayıyı sonraki kurallar okur)
    t = re.sub(r"₺\s*(\d[\d.,]*)", r" \1 lira ", t)
    t = re.sub(r"(\d[\d.,]*)\s*₺", r" \1 lira ", t)
    t = t.replace("₺", " lira ").replace("€", " avro ").replace("$", " dolar ")
    t = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", t)                     # 1.000 -> 1000

    def _pct(m):  # %12,5 dahil — ondalığı kendisi işler (kural-sırası bug'ı düzeltmesi)
        num = m.group(1)
        if "," in num:
            a, b = num.split(",", 1)
            return " yüzde " + _num_tr(int(a)) + " virgül " + _spell_digits(b) + " "
        return " yüzde " + _num_tr(int(num)) + " "
    t = re.sub(r"%\s*(\d+(?:,\d+)?)", _pct, t)
    t = re.sub(r"(\d+(?:,\d+)?)\s*%", _pct, t)
    t = re.sub(r"(\d+),(\d+)",
               lambda m: _num_tr(int(m.group(1))) + " virgül " + _spell_digits(m.group(2)), t)
    t = re.sub(r"(?<=\d)[./:](?=\d)", " ", t)                      # 15.05.2024, 14:30, 7/24

    def _num(m):
        s = m.group(0)
        if len(s) > 4 or (len(s) > 1 and s[0] == "0"):
            return " " + _spell_digits(s) + " "
        return " " + _num_tr(int(s)) + " "

    return re.sub(r"\d+", _num, t)


_lock = threading.Lock()
_model = _tokenizer = _aligner = _dict = None


def _ensure():
    global _model, _tokenizer, _aligner, _dict
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        # macOS Python often lacks system CA certs -> torch.hub download fails SSL.
        try:
            import os, certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except Exception:
            pass
        from torchaudio.pipelines import MMS_FA as bundle
        _model = bundle.get_model().to("cpu").eval()
        _tokenizer = bundle.get_tokenizer()
        _aligner = bundle.get_aligner()
        _dict = bundle.get_dict()


def _words(text):
    """Return (rom_words, orig_words, vis_words): aligner sees romanized letters in
    the dict; original letters + Türkçe-bağlamlı viseme'ler (ğ kuralı) kilitli adımda.
    Metin önce normalize edilir (rakam→okunuş, İ/I, %)."""
    rom_words, orig_words, vis_words = [], [], []
    for raw in normalize_tr(text).split():
        ro, og = [], []
        for ch in raw:
            r = ch.translate(_ROMANIZE)
            if r in _dict and r not in ("-", "'", "*"):
                ro.append(r)
                og.append(ch)
        if ro:
            rom_words.append("".join(ro))
            orig_words.append(og)
            vis_words.append(_word_visemes(og))
    return rom_words, orig_words, vis_words


def align_visemes(pcm: bytes, sr: int, text: str, min_ms: int = 55):
    """Force-align text↔WAV and emit letter-timed viseme cues."""
    _ensure()
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    wav = torch.from_numpy(x)[None, :]
    if sr != _SR:
        import torchaudio
        wav = torchaudio.functional.resample(wav, sr, _SR)
    rom_words, orig_words, vis_words = _words(text)
    if not rom_words:
        return []
    with torch.inference_mode():
        emission, _ = _model(wav)
        token_spans = _aligner(emission[0], _tokenizer(rom_words))
    ratio = wav.size(1) / emission.size(1) / _SR  # emission-frame -> seconds

    # Forced alignment marks each letter's PEAK frame (~20 ms), not its full span.
    # A letter actually lasts until the NEXT letter's onset, so chain by onset time.
    raw = []  # (onset, peak_end, orig_char, viseme_name, rig_id)
    for spans, og, vg in zip(token_spans, orig_words, vis_words):
        for span, ch, (name, vid) in zip(spans, og, vg):
            raw.append((span.start * ratio, span.end * ratio, ch, name, vid))
    if not raw:
        return []
    raw.sort(key=lambda r: r[0])
    tail = max(raw[-1][1], raw[-1][0] + 0.17)     # let the final sound finish, not clip

    cues = []

    def push(off, end, name, vid, ph):
        if end - off <= 0:
            return
        if cues and cues[-1]["id"] == vid:        # merge consecutive same shape
            cues[-1]["end"] = round(end, 3)
        else:
            cues.append({"offset": round(off, 3), "end": round(end, 3),
                         "value": name, "id": vid, "ph": ph})

    if raw[0][0] > 0.04:                           # leading silence -> rest
        push(0.0, raw[0][0], "rest", 0, "_")
    for i, (s, _e, ch, name, vid) in enumerate(raw):
        nxt = raw[i + 1][0] if i + 1 < len(raw) else tail
        gap = nxt - s
        if gap > 0.22:
            # Pause: show the last sound briefly, then CLOSE for the silence. Longer
            # (sentence-level) pauses close sooner so the mouth doesn't linger open
            # between sentences — settle to rest and stay relaxed until the next one.
            hold = 0.06 if gap > 0.45 else 0.09
            push(s, s + hold, name, vid, ch)
            push(s + hold, nxt, "rest", 0, "_")
        else:
            push(s, nxt, name, vid, ch)

    # min-duration: fold sub-min_ms cues into the previous shape (kills jitter).
    # TAVAN: birleşen şekil 0.35s'i aşacaksa kısa cue'yu KORU — TTS bir bölgeyi
    # hızlı/bulanık söylediğinde harfler üst üste binip tek donuk cue oluşuyordu
    # (ölçüldü: 'iki bin yirmi' 0.56s tek II). Kısa ama görünür artikülasyon > donma.
    out = []
    thr = min_ms / 1000.0
    for c in cues:
        if out and (c["end"] - c["offset"]) < thr and (c["end"] - out[-1]["offset"]) <= 0.35:
            out[-1]["end"] = c["end"]
        else:
            out.append(c)
    # re-merge after folding
    merged = []
    for c in out:
        if merged and merged[-1]["id"] == c["id"]:
            merged[-1]["end"] = c["end"]
        else:
            merged.append(dict(c))
    return merged


def split_sentences(text, max_len=140):
    """Metni cümle/öbeklere böl (streaming için). Cümle sonlarında (.!?;:) böler;
    çok uzun cümleleri virgülden kırar; küçük parçaları birleştirir. İlk parça
    kısa olur -> ilk ses çabuk başlar."""
    parts = re.findall(r"[^.!?;:]+[.!?;:]?", text)
    # rakam-içi ayraçta bölme (14:30, 15.05.2024): parçaları geri yapıştır
    fixed = []
    for p in parts:
        if fixed and re.search(r"\d[.:]$", fixed[-1].rstrip()) and re.match(r"\s*\d", p):
            fixed[-1] = fixed[-1] + p
        else:
            fixed.append(p)
    parts = fixed
    pieces = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_len:                       # uzun cümleyi virgülden kır
            sub = re.findall(r"[^,]+,?", p)
            cur = ""
            for s in sub:
                s = s.strip()
                if cur and len(cur) + len(s) > max_len:
                    pieces.append(cur)
                    cur = s
                else:
                    cur = (cur + " " + s).strip()
            if cur:
                pieces.append(cur)
        else:
            pieces.append(p)
    # küçük parçaları (ör. tek kelime) öncekine kat
    out = []
    for p in pieces:
        if out and len(p) < 12:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out or [text.strip()]


def _merge_same(seq):
    """Ardışık aynı viseme'leri tek cue'da birleştir."""
    out = []
    for c in seq:
        if out and out[-1]["id"] == c["id"]:
            out[-1]["end"] = c["end"]
        else:
            out.append(dict(c))
    return out


def smooth_cues(cues, calm):
    """'Hareket azalt' (calm 0-100): en kısa cue'lardan ~calm%'sini önceki ağız
    şekline katarak yutar (DOĞRUSAL his). Ham hizalama bozulmaz."""
    merged = _merge_same(cues)
    if not merged or calm <= 0:
        return merged
    durs = sorted(c["end"] - c["offset"] for c in merged)
    k = int(round(len(durs) * min(100, calm) / 100.0))
    if k <= 0:
        return merged
    thr = durs[min(k - 1, len(durs) - 1)]
    kept = []
    for c in merged:
        if kept and (c["end"] - c["offset"]) <= thr:
            kept[-1]["end"] = c["end"]
        else:
            kept.append(c)
    return _merge_same(kept)


# Portable viseme names (any avatar maps these to its own shapes).
VISEME_SET = {
    "rest": "sessizlik/kapalı", "AA": "a", "E": "e", "II": "i/ı", "O": "o/ö", "U": "u/ü",
    "MBP": "m/b/p", "FV": "f/v", "TD": "t/d/n/r", "SZ": "s/z", "CH": "ş/ç/c/j",
    "KG": "k/g", "L": "l", "H": "h",
}  # not: 'ğ' önceki ünlünün şeklini alır (yumuşak g ünlüyü uzatır), KG'ye gitmez


def finalize(cues, openness=1.0, base_mix=0.09):
    """Raw cues -> FINISHED animation cues (what the avatar plays AND what's downloaded).
    Bakes in: anticipation LEAD (not after a rest), softer PAUSE_MIX at sentence
    edges, and per-shape openness (alpha). Output cue = fully self-describing."""
    n = len(cues)
    if n == 0:
        return []
    # bake the lead into start times (no lead on the first shape after a rest)
    starts = []
    for i, c in enumerate(cues):
        prev_rest = i > 0 and cues[i - 1]["id"] == 0
        starts.append(max(0.0, c["offset"] - (0.0 if prev_rest else LEAD)))
    out = []
    for i, c in enumerate(cues):
        st = starts[i]
        en = starts[i + 1] if i + 1 < n else c["end"]
        if en <= st:
            en = max(c["end"], st + 0.02)
        prev_rest = i > 0 and cues[i - 1]["id"] == 0
        cid = c["id"]
        mix = PAUSE_MIX if (cid == 0 or prev_rest) else base_mix
        out.append({
            "start": round(st, 3), "end": round(en, 3),
            "viseme": c["value"], "rigId": cid,
            "alpha": round(openness * OPEN_MAP.get(cid, 1.0), 3),
            "mix": round(mix, 3), "phoneme": c["ph"],
        })
    return out
