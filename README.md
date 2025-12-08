# 🏠 Sistem Pendukung Keputusan (SPK) Lokasi Ritel Berbasis Geo-Sentimen

## 🚀 Gambaran Umum Proyek

Proyek ini mengimplementasikan Sistem Pendukung Keputusan (SPK) untuk menentukan lokasi optimal pembukaan ritel baru. Keputusan didasarkan pada dua mesin analisis utama yang terintegrasi:

1.  **Geo-spasial Engine (AI):** Menganalisis potensi fisik lokasi (kepadatan, aksesibilitas, pesaing) menggunakan *Clustering* **K-Means** dan data dari **OpenStreetMap (OSM)**.
2.  **Sentiment Engine (AI):** Menganalisis persepsi pasar dan opini publik di area tersebut menggunakan **Natural Language Processing (NLP) VADER**.

Hasil dari kedua mesin AI diintegrasikan menggunakan metode **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** untuk menghasilkan peringkat akhir lokasi yang paling layak.

## 🛠️ Persyaratan dan Dependensi

Pastikan Anda memiliki **Python 3.8+** terinstal.

### A. Dependensi Python

Semua *library* yang diperlukan dapat diinstal menggunakan `pip`.

```bash
pip install Flask pandas numpy scikit-learn nltk geopy osmnx
```

### B. Persyaratan Tambahan (NLTK Data)

Proyek ini membutuhkan *resource* `vader_lexicon` dari NLTK untuk analisis sentimen. Anda perlu mengunduhnya secara manual jika belum ada:

1.  Buka *interpreter* Python:
    ```bash
    python
    ```
2.  Unduh *resource*:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    exit()
    ```

-----

## ⚙️ Cara Menjalankan Proyek

Proyek ini dijalankan sebagai *backend* REST API menggunakan *framework* Flask.

### 1\. Pengaturan Virtual Environment (Venv)

Sangat disarankan untuk menjalankan proyek dalam *virtual environment* terisolasi.

```bash
# Buat venv (lewati jika sudah ada)
python -m venv spk_venv

# Aktivasi venv
# Windows:
spk_venv\Scripts\activate
# macOS/Linux:
source spk_venv/bin/activate

# Lakukan instalasi dependensi Python di langkah sebelumnya.
```

### 2\. Memperbarui Konfigurasi Data AI

Pastikan file-file model Anda telah dikonfigurasi dengan benar:

  * **`models/geo_model.py`:** Pastikan Anda telah menetapkan *Tags* untuk OSMnx (`COMPETITOR_TAGS` dan `SUPPORT_TAGS`) yang sesuai dengan jenis ritel yang Anda analisis.
  * **`spk/topsis_core.py`:** Sesuaikan **`WEIGHTS`** (bobot kepentingan $W_j$) untuk setiap kriteria jika diperlukan.

### 3\. Menjalankan Server Flask

Setelah `venv` aktif dan semua dependensi terinstal, jalankan *file* utama (`app.py`):

```bash
python app.py
```

Server akan mulai berjalan dan *listening* pada: `http://127.0.0.1:5000`

-----

## 💻 Cara Menguji API (Request Postman)

Untuk mendapatkan rekomendasi, Anda harus mengirim *request* **POST** ke *endpoint* `/api/recommend`.

### A. Endpoint

```
Method: POST
URL: http://127.0.0.1:5000/api/recommend
```

### B. Request Body (JSON)

Anda harus mengirim daftar lokasi alternatif yang ingin dianalisis. *Geo-spasial Engine* akan menggunakan `lat` dan `lon` untuk mengambil data dari OSMnx.

```json
{
    "alternatives": {
        "Lokasi_Alternatif_1": {
            "lat": -6.21,
            "lon": 106.82,
            "reviews": ["Tempat strategis tapi ramai sekali", "Banyak pilihan"],
            "competitor_dist": 200, 
            "sewa_cost": 100
        },
        "Lokasi_Alternatif_2": {
            "lat": -6.50,
            "lon": 106.70,
            "reviews": ["Jauh, tapi pelayanannya bagus", "harga sangat terjangkau"],
            "competitor_dist": 800,
            "sewa_cost": 50
        }
    }
}
```

### C. Response (Contoh Output)

Server akan merespons dengan status `200 OK` dan *ranking* akhir:

```json
{
    "status": "success",
    "message": "Rekomendasi lokasi berhasil dihitung.",
    "ranking": [
        {
            "id": "Lokasi_Alternatif_1",
            "C1_Geo": 0.92,
            "C2_Sentiment": 0.78,
            "C3_CompetitorDist": 200,
            "C4_Sewa": 100,
            "Vi": 0.65,
            "Ranking": 1
        },
    ]
}
```

-----

## 📂 Struktur Proyek

```
TopsisSPPK/
├── app.py                   # Backend utama Flask & Routing
├── models/
│   ├── geo_model.py         # Geo-spasial Engine (K-Means & OSMnx)
│   └── sentiment_model.py   # Sentiment Engine (NLP VADER)
├── spk/
│   └── topsis_core.py       # SPK Core (Implementasi TOPSIS)
├── spk_venv/                # Virtual Environment
└── README.md                # Dokumentasi Proyek ini
```

