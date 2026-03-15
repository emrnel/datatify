# Datatify

**Spotify dinleme geçmişinden kişisel davranış analitiği.**

Datatify, Spotify Genişletilmiş Akış Geçmişi verilerini yükleyerek dinleme alışkanlıklarını 25'ten fazla davranışsal metrikle analiz eden, yapay zeka destekli bir web uygulamasıdır.

🌐 **Canlı Demo:** [https://web-production-65055.up.railway.app](https://web-production-65055.up.railway.app)

---

## Özellikler

### Davranışsal Analiz (25+ Metrik)
- **Sabırsızlık & Dikkat:** Erken atlama oranı, parça tamamlama yüzdesi, atlama gecikmesi
- **Çeşitlilik & Keşif:** Sanatçı çeşitlilik skoru (Shannon entropisi), müzik yenilik oranı
- **Alışkanlık & Rutin:** Sirkadiyen dinleme profili, gece dinleme oranı, alışkanlık döngüsü skoru
- **Sadakat & Odak:** Sanatçı sadakat skoru, şarkı tekrar yoğunluğu, odak oturumu skoru
- **Platform Bağlamı:** Mobil kullanım, çevrimdışı oran, karıştırma oranı

### Oyunlaştırma
- **15 Açılabilir Rozet** (Gece Kuşu, Maraton Dinleyici, Kaşif, Yüzyılcı...)
- **10 Seviyeli İlerleme Sistemi** (Acemi → Aşkın)
- **Arketip Sınıflandırması** (Gece Dalgıcı, Huzursuz Kaşif, Sadık Bekçi...)
- **6 Boyutlu Radar Grafik** Profili: Sabır, Keşif, Sadakat, Odak, Çeşitlilik, Gece Kuşu

### Yapay Zeka Analizi
- Google Gemini 2.0 Flash ile kişilik profili çıkarımı
- Psikolojik analiz, kişilik özellikleri ve veri destekli içgörüler

### Topluluk Karşılaştırması
- Anonim metrik gönderimi
- Diğer kullanıcılara göre yüzdelik sıralama

---

## Kurulum

### Gereksinimler
- Python 3.9+
- Google Gemini API anahtarı ([buradan alın](https://aistudio.google.com))

### Adımlar

```bash
# Repoyu klonlayın
git clone https://github.com/emrnel/datatify
cd datatify

# Sanal ortam oluşturun
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Ortam değişkenini ayarlayın
export GEMINI_API_KEY="your-api-key-here"

# Sunucuyu başlatın
python main.py
```

Tarayıcıda `http://localhost:8000` adresini açın.

### Ortam Değişkenleri

| Değişken | Açıklama | Zorunlu |
|----------|----------|---------|
| `GEMINI_API_KEY` | Google Gemini API anahtarı | Evet (AI analizi için) |
| `SKIP_GEMINI` | `1` yaparsanız AI analizi atlanır | Hayır |
| `DB_PATH` | SQLite veritabanı yolu | Hayır |
| `PORT` | Sunucu portu (varsayılan: 8000) | Hayır |

---

## Spotify Verisi Nasıl İndirilir?

1. [Spotify Hesap Sayfasına](https://www.spotify.com/account/privacy/) gidin
2. **"Genişletilmiş Akış Geçmişi"** bölümünden veri isteğinde bulunun
3. ~30 gün içinde e-posta ile gelen ZIP dosyasını açın
4. `Streaming_History_Audio_*.json` dosyalarını Datatify'a yükleyin

---

## API Endpointleri

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Ana sayfa |
| `POST` | `/analyze` | JSON dosyası yükle ve analiz et |
| `POST` | `/api/submit` | Anonim metrik gönder |
| `POST` | `/api/percentiles` | Yüzdelik sıralama al |
| `GET` | `/api/stats` | Topluluk istatistikleri |
| `GET` | `/api/health` | Sağlık kontrolü |

---

## Teknoloji Yığını

| Katman | Teknoloji |
|--------|-----------|
| Backend | FastAPI + Uvicorn |
| Yapay Zeka | Google Gemini 2.0 Flash |
| Veritabanı | SQLite |
| Frontend | HTML5 / CSS3 / JavaScript |
| Grafikler | Chart.js |
| Deploy | Railway (NIXPACKS) |

---

## Deployment

Railway üzerinde otomatik deploy için:

1. Bu repoyu Railway'e bağlayın
2. `GEMINI_API_KEY` ortam değişkenini ekleyin
3. `railway.json` yapılandırması otomatik uygulanır

---

## Lisans

MIT
