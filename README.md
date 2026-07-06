# Vislign

*Turkish lip-sync / viseme generator: give it audio + the spoken text, get an
accurate, ready-to-play mouth-shape timeline via forced alignment. Works with
any TTS engine. Docs below are in Turkish.*

**Türkçe için özelleştirilmiş bir viseme (ağız-şekli) üreteci.** Metin ve sesi
verirsin, sana o sesin her anında ağzın hangi şekilde olması gerektiğini
söyleyen bir zaman çizelgesi döner. Konuşan avatar / dijital insan / dublaj
karakteri gibi işlerde dudak senkronu için kullanılır.

## Neden var

Klasik lip-sync araçları (Rhubarb gibi) sadece **sese bakarak** ağız şekli
tahmin eder — metni hiç görmez, kör bir tahmindir. Elimizde zaten hem metin
hem ses varsa (bir TTS'ten geçtiği için) bu tahmine hiç gerek yok: metni ve
sesi **hizalayıp** her harfin gerçekte ne zaman söylendiğini doğrudan
öğrenebiliriz. Sonuç çok daha isabetli bir dudak senkronu — özellikle
Türkçe'de, çünkü Türkçe'ye özel kurallar (aşağıda) da işin içine giriyor.

## Girdi

Vislign bir **TTS değil** — kendi başına ses üretmez. İki şey verirsin:

1. **Ses** — herhangi bir TTS'ten (ya da kayıttan) çıkmış, zaten hazır bir WAV.
2. **Metin** — o seste tam olarak söylenen cümle.

Ve sana o sesin zaman çizelgesinde ağzın ne zaman ne şekilde olması gerektiğini
döner. Yani hangi TTS'i kullanırsan kullan (yerel/bulut, hangi model olursa
olsun) — Vislign'ı sesin üstüne aynen takabilirsin.

## Nasıl çalışır (kısaca)

1. **Metni normalize et** — TTS'in gerçekte SÖYLEDİĞİ kelimelere çevir:
   rakamlar ("2024" → "iki bin yirmi dört"), ondalıklar ("41,73" → "kırk bir
   virgül yetmiş üç"), sıra sayıları ("2. adım" → "ikinci adım"), birimler
   ("10 dk" → "on dakika"), kısaltmalar ("KVKK" → "ka ve ka ka"), e-posta/URL
   ("@" → "et", alan noktası → "nokta"), %/tarih/saat/₺ işaretleri ve Türkçe
   büyük/küçük harf kuralları (İ/I).
2. **Metin + sesi hizala** — açık kaynak bir konuşma-hizalama modeli
   (`torchaudio`'nun MMS Forced Aligner'ı) her harfin sesin içinde tam olarak
   nerede başlayıp bittiğini çıkarır.
3. **Harfleri ağız şekline eşle** — Türkçe fonetik kurallarıyla 14 ağız
   şekline (a, e, i/ı, o/ö, u/ü, kapalı dudak, vb.) eşlenir. Örneğin yumuşak
   **ğ** kendi başına şekil almaz, kendinden önceki ünlüyü uzatır — gerçek
   ağız da böyle davranır.
4. **Titremeyi/donmayı temizle, duraklamayı sesle kapat** — çok kısa şekilleri
   toparlayarak titremeyi, çok uzun donuk şekilleri bölerek donmayı önler.
   Duraklamalarda ağız sabit bir zamanlayıcıyla değil, **ses gerçekten
   kesildiğinde** kapanır (enerji ölçümlü) — kelime-sonu uzayan hecelerde ağız
   konuşurken kapanmaz, cümle sonunda ise hızlıca kapanır.

Çıktı, oynatmaya hazır bir zaman çizelgesi (JSON) — istediğin avatar/rig'e
bağlayabilirsin.

## Kurulum

```bash
pip install -r requirements.txt
```

İlk çalıştırmada hizalama modeli (~1.2 GB) otomatik indirilir ve
`~/.cache`'e kaydedilir; sonraki çalıştırmalar için tekrar inmez.

## Hızlı kullanım

```python
from vislign import align_visemes, finalize
import soundfile as sf

data, sr = sf.read("ornek.wav", dtype="int16", always_2d=True)
pcm = data[:, 0].tobytes()

cues = finalize(align_visemes(pcm, sr, "Merhaba, nasılsın?"))
```

Ya da komut satırından:

```bash
python example.py ornek.wav "Merhaba, nasılsın?"
```

### Cümle + kelime zaman çizelgesi (altyazı/karaoke)

Aynı hizalama geçişinden **cümle ve kelime zamanları** da alınabilir — viseme'ler
ve sesle örnek-düzeyinde tutarlıdır (altyazı vurgulama, karaoke, kelime-kelime
highlight için):

```python
from vislign import align_visemes_ex, finalize

ex = align_visemes_ex(pcm, sr, "Alışkanlıklarınız kimliğinizi nasıl şekillendirir?")
cues = finalize(ex["cues"])
ex["sentences"]
# [{"id": 1, "text": "Alışkanlıklarınız kimliğinizi nasıl şekillendirir?",
#   "start": 0.64, "end": 3.77,
#   "words": [{"id": 1, "text": "Alışkanlıklarınız", "start": 0.64, "end": 1.77},
#             {"id": 2, "text": "kimliğinizi",       "start": 1.77, "end": 2.47},
#             ...]}]
```

Kelime metinleri **orijinal** yazımıyla döner ("2028'de", "%50", "2. adım" gibi
normalize edilen ifadeler dahil) — zamanlar, normalize edilip hizalanan
karşılıklarından geri eşlenir.

## Çıktı formatı

`finalize()` her biri kendi kendini anlatan cue'lardan oluşan **bir liste**
döner — "Merhaba" sesinin gerçek çıktısından bir kesit:

```json
[
  {"start": 0.363, "end": 0.383, "viseme": "MBP", "rigId": 21, "alpha": 1.0,  "mix": 0.15, "phoneme": "m"},
  {"start": 0.383, "end": 0.524, "viseme": "E",   "rigId": 4,  "alpha": 1.0,  "mix": 0.09, "phoneme": "e"},
  {"start": 0.524, "end": 0.585, "viseme": "AA",  "rigId": 2,  "alpha": 0.72, "mix": 0.09, "phoneme": "a"},
  {"start": 0.585, "end": 0.666, "viseme": "MBP", "rigId": 21, "alpha": 1.0,  "mix": 0.09, "phoneme": "b"},
  {"start": 0.666, "end": 0.756, "viseme": "AA",  "rigId": 2,  "alpha": 0.72, "mix": 0.09, "phoneme": "a"}
]
```

| Alan | Anlamı |
|---|---|
| `start` / `end` | Bu ağız şeklinin ne zaman gösterileceği (saniye) |
| `viseme` | Taşınabilir şekil adı — herhangi bir rig'e eşlenebilir |
| `rigId` | Bu repodaki örnek rig'in animasyon id'si (kendi rig'inde yok sayılabilir) |
| `alpha` | Ağzın ne kadar açık olacağı (0-1) |
| `mix` | Bir sonraki şekle geçiş süresi (saniye, blend/crossfade) |
| `phoneme` | Kaynak harf (bilgi amaçlı) |

## Viseme seti

| Şekil | Türkçe ses |
|---|---|
| `rest` | sessizlik |
| `AA` | a |
| `E` | e |
| `II` | i, ı |
| `O` | o, ö |
| `U` | u, ü |
| `MBP` | m, b, p |
| `FV` | f, v |
| `TD` | t, d, n, r |
| `SZ` | s, z |
| `CH` | ş, ç, c, j |
| `KG` | k, g |
| `L` | l |
| `H` | h |

## Bir avatara bağlama

Aldığın `cues` listesini avatarında oynatmak iki basit kurala iniyor:

1. **Ses saatine kilitle.** `setTimeout` gibi ayrı bir zamanlayıcı kullanma —
   sesin gerçek oynatma anını (ör. Web Audio'da `audioContext.currentTime`)
   her karede oku, `cue.start`'ı geçen cue'ları sırayla uygula.
2. **Şekli bir sonraki cue'ya kadar tut.** Her cue bitene kadar değil,
   **sıradaki cue gelene kadar** o ağız şeklinde kal (cue'lar zaten bitişik).

```js
// t = sesin şu anki oynatma zamanı (saniye)
while (cues[i] && cues[i].start <= t) {
  setMouthShape(cues[i].viseme, cues[i].alpha, cues[i].mix); // kendi rig'ine göre
  i++;
}
```

`setMouthShape(viseme, alpha, mix)` kendi rig'ine göre yazacağın küçük bir
fonksiyon: `viseme` adını kendi ağız şekline eşle, `alpha` ile ne kadar açık
olacağını, `mix` ile bir önceki şekilden ne kadar sürede geçiş yapacağını
(crossfade) ayarla. **`alpha`/`mix` zaten bitmiş halde geliyor — üstüne
kendi yumuşatmanı ekleme**, aksi halde hareket beklenenden daha yavaş/donuk
görünür.

Ses ile senkron kalması için, sesi ve cue döngüsünü **aynı anda** başlat.

## Sınırlamalar

- Şu an için **yalnızca Türkçe** özelleştirilmiş (harf→şekil tablosu ve metin
  normalizasyonu Türkçe'ye göre). Başka bir dile taşımak için `vislign.py`
  içindeki `_VIS` tablosunu ve `normalize_tr()` fonksiyonunu o dile göre
  uyarlaman gerekir.
- Hizalama modeli (`torchaudio` MMS Forced Aligner) Meta'nın MMS projesine
  dayanıyor ve **CC-BY-NC 4.0** (ticari olmayan kullanım) lisansıyla
  dağıtılıyor. Bu repodaki kod MIT lisanslı, ama ticari bir üründe
  kullanmadan önce model ağırlığının lisansını kontrol et.

## Lisans

Kod MIT lisanslıdır — bkz. `LICENSE`.
