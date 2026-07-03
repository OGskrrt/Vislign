# CLAUDE.md

Bu repo **Vislign** — Türkçe metin+ses için viseme (ağız-şekli) zaman
çizelgesi üreten, tek dosyalık, bağımsız bir Python kütüphanesi. Detaylı
açıklama için `README.md`'ye bak; burası Claude'un bu repoyu hızlı
kavraması için kısa bir özet.

## Ne yapar, ne yapmaz

- **Yapar:** verilen bir WAV dosyası + o WAV'da söylenen metni hizalar,
  her harfin sesin içinde tam olarak ne zaman geçtiğini bulur, Türkçe'ye
  özel kurallarla 14 ağız şekline (viseme) çevirir.
- **Yapmaz:** ses üretmez (TTS değildir). Kullanıcı sesi zaten başka bir
  yerden (herhangi bir TTS motoru, kayıt, vs.) getirmiş olmalı.

## Tek dosya: `vislign.py`

Kullanılacak asıl iki fonksiyon:

```python
from vislign import align_visemes, finalize

raw_cues = align_visemes(pcm_bytes, sample_rate, "Söylenen tam metin")
cues = finalize(raw_cues)  # oynatmaya hazır, bitmiş cue listesi
```

- `pcm_bytes`: mono, 16-bit little-endian PCM (ör. `soundfile` ile okunan
  WAV'ın `.tobytes()` hali). `align_visemes` ihtiyaç olursa 16 kHz'e kendi
  resample eder.
- `align_visemes` içeride `torchaudio`'nun MMS Forced Aligner modelini
  **ilk çağrıda** indirir (~1.2 GB, `~/.cache` altına) ve önbelleğe alır.
- `finalize(cues, openness=1.0, base_mix=0.09)`: ham cue'ları oynatmaya
  hazır hale getirir (lead + duraklama yumuşatma + açıklık gömülü).

Yardımcı fonksiyonlar: `smooth_cues(cues, calm)` (0-100, gereksiz kısa
cue'ları komşusuna katarak hareketi sadeleştirir — **HAM cue'lar üzerinde,
yani `finalize`'dan ÖNCE çağrılır**; finalized cue'lara uygulanırsa KeyError
verir), `split_sentences(text)`
(uzun metni cümlelere böler — streaming/parça-parça TTS senaryosunda
kullanışlı), `normalize_tr(text)` (yalnız hizalayıcının gördüğü metni
normalize eder; TTS'e giden asıl metni DEĞİŞTİRMEZ — rakamlar, ondalıklar,
sıra sayıları "2."→"ikinci", birimler "dk"→"dakika", kısaltmalar, e-posta/URL
"@"→"et"/"nokta", %/tarih/saat/₺, İ/I küçültme).

## Bir projeye entegre ederken

1. `pip install -r requirements.txt`.
2. Kullanıcının TTS'inden (Claude'a ne kullandığını sorabilirsin) WAV +
   metin al.
3. `finalize(align_visemes(pcm, sr, text))` çağır, `cues` listesini avatar
   oynatma döngüsüne ver.
4. Oynatma tarafında (frontend/oyun motoru neyse): ses saatine kilitli bir
   döngüde, her cue'yu **bir sonraki cue gelene kadar** göster. `alpha`/
   `mix` zaten bitmiş halde geliyor, üstüne ekstra yumuşatma **ekleme**.
   Ayrıntı ve kod örneği için `README.md` → "Bir avatara bağlama".
5. `viseme` alanı taşınabilir şekil adıdır (AA/E/II/O/U/MBP/FV/TD/SZ/CH/
   KG/L/H/rest) — hedef avatarın kendi şekillerine bir kez eşle. `rigId`'yi
   yok sayabilirsin, bu repodaki referans rig'e özel.

## Dikkat edilmesi gerekenler

- Bu kütüphane **yalnızca Türkçe** için kalibre edilmiştir (harf→şekil
  tablosu + metin normalizasyonu). Başka bir dil isteniyorsa `_VIS`
  tablosu ve `normalize_tr()` o dile göre yeniden yazılmalı — otomatik
  çok-dilli değildir.
- Hizalama modeli **CC-BY-NC 4.0** (ticari olmayan kullanım) lisanslıdır.
  Kullanıcı ticari bir üründen bahsediyorsa bunu ona hatırlat.
- Model indirmesi büyük (~1.2 GB) ve ilk çağrıda birkaç saniye sürer;
  sonraki çağrılar hızlıdır (modele bir kez yüklenir, süreç boyunca
  bellekte kalır).
