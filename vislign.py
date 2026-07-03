"""
Vislign — text + audio -> portable visemes via FORCED ALIGNMENT (torchaudio MMS_FA).

Acoustic lip-sync tools (Rhubarb and friends) guess mouth shapes from the audio
waveform alone — they never see the text, so they're blind and limited to a
handful of shapes. Vislign already knows the text (you fed it to your TTS), so
it force-aligns text<->audio and reads each letter's real timing straight out of
the alignment model. That gives much more accurate, much richer lip-sync — and
because it's just text+audio, it works with ANY TTS engine, not a specific one.

This build ships a Turkish letter->viseme table and Turkish-specific text
normalization (numbers, ordinals, units, abbreviations, %, dates/times, e-mail,
İ/I casing, ğ handling). Porting to another language means swapping the `_VIS`
table and `normalize_tr`.

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

# Turkish letter -> (portable viseme name, this-rig viseme id). Rig calibration:
# 0=rest 2=a 4=e 6=i/ı(wide) 7=u/ü(round-narrow) 8=o/ö(round-open) 21=closed(MBP)
# 18=F/V 19=T/D/N/R 15=S/Z 16=Ş/Ç/C/J 20=K/G/Ğ 14=L 12=H
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

# Sayı sonrası birimler (TTS "10 dk"yı "on dakika" okur; hizalama da öyle görmeli —
# ölçüldü: "dk" harf-harf hizalanınca 'dakika'nın 3 ünlüsü boyunca ağız kilitleniyordu).
_UNITS = {"km": "kilometre", "cm": "santimetre", "mm": "milimetre", "kg": "kilogram",
          "gr": "gram", "gb": "gigabayt", "mb": "megabayt", "tb": "terabayt",
          "kb": "kilobayt", "ml": "mililitre", "lt": "litre", "dk": "dakika",
          "sn": "saniye", "sa": "saat", "tl": "lira", "m": "metre", "g": "gram", "l": "litre"}

# Sıra sayıları: kardinalin SON kelimesi sıralı biçime döner ("yirmi beş" -> "yirmi beşinci").
_ORD_LAST = {"bir": "birinci", "iki": "ikinci", "üç": "üçüncü", "dört": "dördüncü",
             "beş": "beşinci", "altı": "altıncı", "yedi": "yedinci", "sekiz": "sekizinci",
             "dokuz": "dokuzuncu", "on": "onuncu", "yirmi": "yirminci", "otuz": "otuzuncu",
             "kırk": "kırkıncı", "elli": "ellinci", "altmış": "altmışıncı",
             "yetmiş": "yetmişinci", "seksen": "sekseninci", "doksan": "doksanıncı",
             "yüz": "yüzüncü", "bin": "bininci"}


def _ord_tr(n: int) -> str:
    """1..99 -> Türkçe sıra sayısı okunuşu ("2."->"ikinci", "25."->"yirmi beşinci")."""
    w = _num_tr(n).split()
    w[-1] = _ORD_LAST.get(w[-1], w[-1])
    return " ".join(w)


def _frac_tr(b: str) -> str:
    """Ondalık kesir okunuşu: TTS "41,73"ü "kırk bir virgül YETMİŞ ÜÇ" okur (kardinal),
    rakam-rakam değil — ölçüldü: rakam-rakam hizalama 0.6s+ sesli-kapalı üretiyordu.
    Baş-sıfırlı kesirler ("0,05") rakam-rakam kalır ("sıfır beş")."""
    return _spell_digits(b) if b.startswith("0") else _num_tr(int(b))


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
    # e-posta/URL: TTS "@"yı "et", alan-adı noktasını "nokta" okur (ölçüldü) —
    # hizalama da aynı kelimeleri görmeli, yoksa adres içinde ağız kapalı kalıyor.
    t = t.replace("@", " et ")
    t = re.sub(r"(?<=[a-zçğıöşü0-9])\.(?=[a-zçğıöşü]{2,})", " nokta ", t)
    # sıra sayıları: "2. adım" -> "ikinci adım" (TTS böyle okur; 1-2 hane — yıllar hariç)
    t = re.sub(r"\b(\d{1,2})\.(?=\s)", lambda m: " " + _ord_tr(int(m.group(1))) + " ", t)
    # sayı+birim: "10 dk" -> "10 dakika" (TTS birimi açar; sonra sayı kuralı okur)
    t = re.sub(r"(\d)\s*(km|cm|mm|kg|gr|gb|mb|tb|kb|ml|lt|dk|sn|sa|tl|m|g|l)\b",
               lambda m: m.group(1) + " " + _UNITS[m.group(2)] + " ", t)
    # para: ₺150 / 150₺ -> "... lira" (sayıyı sonraki kurallar okur)
    t = re.sub(r"₺\s*(\d[\d.,]*)", r" \1 lira ", t)
    t = re.sub(r"(\d[\d.,]*)\s*₺", r" \1 lira ", t)
    t = t.replace("₺", " lira ").replace("€", " avro ").replace("$", " dolar ")
    t = re.sub(r"(?<=\d)\.(?=\d{3}\b)", "", t)                     # 1.000 -> 1000

    def _pct(m):  # %12,5 dahil — ondalığı kendisi işler (kural-sırası bug'ı düzeltmesi)
        num = m.group(1)
        if "," in num:
            a, b = num.split(",", 1)
            return " yüzde " + _num_tr(int(a)) + " virgül " + _frac_tr(b) + " "
        return " yüzde " + _num_tr(int(num)) + " "
    t = re.sub(r"%\s*(\d+(?:,\d+)?)", _pct, t)
    t = re.sub(r"(\d+(?:,\d+)?)\s*%", _pct, t)
    t = re.sub(r"(\d+),(\d+)",
               lambda m: _num_tr(int(m.group(1))) + " virgül " + _frac_tr(m.group(2)), t)
    t = re.sub(r"(?<=\d)[./:](?=\d)", " ", t)                      # 15.05.2024, 14:30, 7/24

    def _num(m):
        s = m.group(0)
        # rakam-rakam yalnız telefon-benzeri diziler: baş-sıfırlı ("0850") ya da ≥8 hane.
        # 5-7 haneli çıplak sayılar fiyat/miktar olarak okunur ("54999" -> "elli dört bin
        # dokuz yüz doksan dokuz") — ölçüldü: rakam-rakam verilince hizalama 0.64s geriliyordu.
        if (len(s) > 1 and s[0] == "0") or len(s) >= 8:
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


def align_visemes(pcm: bytes, sr: int, text: str, min_ms: int = 20):
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

    # Enerji zarfı (16k): duraklamada ağzı SES GERÇEKTEN KESİLİNCE kapat. Sabit
    # 60-90ms tutuş, kelime-sonu uzayan hecelerde 0.4-0.7s "konuşurken kapalı ağız"
    # bırakıyordu (ölçüldü). Sessizlik = p90 enerjinin %18'i altı.
    x16 = wav.squeeze(0).cpu().numpy()
    _w = max(1, int(_SR * 0.02))
    _n16 = len(x16) // _w
    env16 = (np.sqrt((x16[:_n16 * _w].reshape(_n16, _w) ** 2).mean(axis=1) + 1e-9)
             if _n16 > 1 else np.zeros(1, dtype=np.float32))
    sil_thr = max(float(np.percentile(env16, 90)) * 0.18, 1e-4)
    spk_thr = max(float(np.percentile(env16, 90)) * 0.30, 2e-4)

    def _first_sil(t0, t1, k=6):
        """t0-t1 içinde en az k ardışık pencere (k*20ms) SÜREKLİ sessizliğin başlangıcı.
        Tek pencerelik çukurlar (grup arası nefes/dip) kapanış tetiklemesin —
        ölçüldü: kısa dipte kapanan ağız hemen ardından gelen seste kapalı kalıyordu."""
        a = max(int(t0 / 0.02), 0)
        b = min(int(t1 / 0.02), _n16)
        run = 0
        for wi in range(a, b):
            if env16[wi] < sil_thr:
                run += 1
                if run >= k:
                    return (wi - k + 1) * 0.02
            else:
                run = 0
        if 0 < run < k and b == _n16:      # ses dosyanın sonunda sessizlik kısa kalmış
            return (b - run) * 0.02
        return None

    def _first_speech(t0, t1):
        a = max(int(t0 / 0.02), 0)
        b = min(int(t1 / 0.02), _n16)
        for wi in range(a, b):
            if env16[wi] >= spk_thr:
                return wi * 0.02
        return None

    # ONSET ÖNE ÇEKME: hizalayıcı sayı/hızlı bölgelerde sonraki kelimenin harflerini
    # GEÇ yerleştirebiliyor -> duraklamadaki sessizlik molasından sonra ses geri
    # başlıyor ama ağız kapalı kalıyordu (ölçüldü: 0.5s+). Ses geri başladığı anda
    # sıradaki harfin onset'ini oraya çek — ağız sesle birlikte açılır.
    for i in range(len(raw) - 1):
        s, nxt = raw[i][0], raw[i + 1][0]
        if nxt - s > 0.22:
            sil = _first_sil(s + 0.06, nxt - 0.04)
            if sil is not None:
                res = _first_speech(sil + 0.04, nxt - 0.04)
                if res is not None and nxt - res > 0.06:
                    o = raw[i + 1]
                    raw[i + 1] = (res, o[1], o[2], o[3], o[4])

    # let the final sound finish, but never exceed the actual audio (players held a
    # stale shape ~130ms past end-of-audio — measured, deterministic). Aligner'ın son
    # onset'i her zaman ses içindedir -> audio_dur tavanı güvenli, +taban GEREKMEZ
    # (önceki +0.05 taban guard'ı 30ms taşma yaratıyordu — ölçüldü).
    audio_dur = (len(pcm) // 2) / sr
    tail = min(max(raw[-1][1], raw[-1][0] + 0.17), audio_dur)
    if tail <= raw[-1][0]:
        tail = raw[-1][0] + 0.02

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
        # aligner çok hızlı bölgede iki harfi AYNI frame'e koyabiliyor -> süre 0 olur
        # ve push tamamen atlardı (ölçüldü: 'com nokta tr' kuyruğunda 5-7 harf birden
        # yutuldu). 25ms taban ver (3-hane yuvarlama 20ms'yi 0.019'a düşürüp katlamaya
        # yakalatabiliyordu); finalize monotonluğu ayrıştırır.
        if nxt <= s:
            nxt = s + 0.025
        gap = nxt - s
        if gap > 0.22:
            # Duraklama: son sesi SES KESİLENE KADAR tut, sonra kapat (enerji-ölçümlü).
            # Cümle sonunda ses hızla kesilir -> hızlı kapanış korunur; kelime-sonu
            # uzayan hecede ses sürer -> ağız konuşurken kapanmaz. Donma tavanı 0.40s.
            sil = _first_sil(s + 0.06, nxt - 0.04)
            if sil is None and gap <= 0.44:
                push(s, nxt, name, vid, ch)          # ses hiç kesilmiyor: kapatma
            else:
                hold_end = sil if sil is not None else s + 0.40
                # 0.42 mutlak tavan: sessizlik geç başlarsa bile tek şekil donma
                # eşiğini (0.45) aşmasın
                hold_end = min(max(hold_end, s + 0.06), nxt - 0.02, s + 0.42)
                push(s, hold_end, name, vid, ch)
                push(hold_end, nxt, "rest", 0, "_")
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


import re


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
    # monotonluk: rest-sonrası cue (lead'siz) ile sıradaki cue (lead'li) AYNI anda
    # başlayabiliyordu -> oynatıcıda ilk şekil 0ms'de eziliyordu (ölçüldü: 'dün' d+ü
    # üst üste). Her start bir öncekinden en az 20ms sonra.
    for i in range(1, n):
        if starts[i] < starts[i - 1] + 0.02:
            starts[i] = starts[i - 1] + 0.02
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
    # kuyruk clamp'i: monotonluk kaydırması kalabalık sonlarda cue'ları ses süresinin
    # ötesine itebiliyordu (ölçüldü ~20-60ms). Geriye doğru sıkıştır — ham cue'ların
    # sonu (align_visemes zaten ses süresine clamp'li) tavandır.
    cap = cues[-1]["end"]
    for c in reversed(out):
        if c["end"] > cap:
            c["end"] = round(cap, 3)
        if c["start"] >= c["end"]:
            c["start"] = round(max(0.0, c["end"] - 0.02), 3)
        cap = c["start"]
    return out


if __name__ == "__main__":
    import sys, soundfile as sf, json
    wav_path, text = sys.argv[1], sys.argv[2]
    data, sr = sf.read(wav_path, dtype="int16", always_2d=True)
    mono = data[:, 0].astype("<i2").tobytes()
    cues = finalize(align_visemes(mono, sr, text))
    print(f"{len(cues)} cues over {cues[-1]['end'] if cues else 0:.2f}s")
    print(json.dumps(cues[:18], ensure_ascii=False, indent=0))
