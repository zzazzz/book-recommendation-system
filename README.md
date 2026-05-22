# 📚 BookMind — Dashboard Sistem Rekomendasi Buku

> Dashboard interaktif berbasis **Streamlit** untuk eksplorasi dataset Book-Crossing, evaluasi model rekomendasi, dan simulasi rekomendasi Top-10 secara real-time.

---

## ✨ Fitur Utama

| Halaman | Deskripsi |
|---|---|
| **🏠 Overview** | Statistik dataset, KPI model terbaik, buku populer, dan visualisasi bobot sinyal hybrid |
| **📊 Data** | Distribusi rating, tren tahun publikasi, penerbit teratas, aktivitas user & buku |
| **📈 Evaluasi** | Radar chart multi-metrik, bar chart perbandingan, tabel metrik lengkap |
| **🤖 Rekomendasi** | Simulasi Top-10 berdasarkan User atau berdasarkan kemiripan Buku |
| **📖 Metodologi** | Penjelasan pipeline data, arsitektur 5 sinyal, dan protokol evaluasi |

---

## 🧠 Arsitektur Model

Model final adalah **Tuned Hybrid** yang menggabungkan lima sinyal:

```
Hybrid Score = 0.00 × Popularity
             + 0.20 × Content-Based (TF-IDF)
             + 0.20 × Co-Like Item KNN
             + 0.20 × Author Affinity
             + 0.40 × User-Based CF
```

### Detail Sinyal

- **Bayesian Popularity** — Skor popularitas dengan prior Bayesian untuk mereduksi bias buku dengan sedikit rating.
- **Content-Based TF-IDF** — Representasi teks judul + penulis + penerbit di-embed dengan TF-IDF (max 20k fitur, bigram). Kemiripan dihitung via cosine similarity.
- **Co-Like Item KNN** — Item-KNN berbasis matriks co-like (rating ≥ 7). Menggunakan `NearestNeighbors` dengan metrik cosine.
- **Author Affinity** — Akumulasi preferensi penulis dari riwayat rating user, dibobot dengan rating relatif.
- **User-Based CF** — Nearest neighbor antar user pada matriks user-item liked (rating ≥ 7), aggregasi skor dari k=80 tetangga terdekat.

---

## 📊 Hasil Evaluasi

Evaluasi dilakukan pada **250 pengguna test set** (Top-10 recommendations):

| Model | Precision@10 | Recall@10 | MAP@10 | NDCG@10 | Diversity | Coverage |
|---|---|---|---|---|---|---|
| **Tuned Hybrid** ⭐ | 0.0184 | 0.184 | 0.0853 | 0.1082 | 0.806 | 38.9% |
| User-Based CF | 0.0152 | 0.152 | 0.0687 | 0.0878 | 0.909 | 20.9% |
| Content-Based | 0.0148 | 0.148 | 0.0560 | 0.0773 | 0.709 | 45.8% |
| Co-Like ItemKNN | 0.0136 | 0.136 | 0.0649 | 0.0813 | 0.916 | 39.6% |
| Author Affinity | 0.0116 | 0.116 | 0.0394 | 0.0571 | 0.764 | 37.2% |
| Popularity | 0.0008 | 0.008 | 0.0027 | 0.0039 | 0.719 | 0.7% |

---

## 📁 Struktur File

```
├── dashboard.py        # Aplikasi Streamlit utama
├── Books.csv           # Metadata buku (ISBN, judul, penulis, penerbit, tahun, URL cover)
├── Ratings.csv         # Rating pengguna (User-ID, ISBN, Book-Rating 0–10)
├── Users.csv           # Data pengguna (User-ID, lokasi, usia)
└── README.md           # Dokumentasi ini
```

> **Catatan:** File CSV tidak termasuk dalam repositori ini karena ukurannya. Dataset dapat diunduh dari [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python 3.9+
- pip

### 2. Instalasi Dependensi

```bash
pip install streamlit pandas numpy scipy scikit-learn plotly
```

Atau menggunakan `requirements.txt`:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
streamlit>=1.30.0
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
plotly>=5.18.0
```

### 3. Siapkan Dataset

Letakkan ketiga file CSV di direktori yang sama dengan `dashboard.py`:

```
Books.csv
Ratings.csv
Users.csv
dashboard.py
```

### 4. Jalankan Dashboard

```bash
streamlit run dashboard.py
```

Buka browser di `http://localhost:8501`

---

## ⚙️ Konfigurasi

Beberapa konstanta dapat disesuaikan di bagian atas `dashboard.py`:

```python
TOP_K = 10                        # Jumlah rekomendasi yang ditampilkan
RELEVANT_THRESHOLD = 7            # Ambang rating untuk item "relevan"
MIN_USER_EXPLICIT_RATINGS = 10    # Minimum rating eksplisit per user (warm filter)
MIN_BOOK_EXPLICIT_RATINGS = 10    # Minimum rating eksplisit per buku (warm filter)
CANDIDATE_POOL = 400              # Pool kandidat sebelum re-ranking
CO_LIKE_NEIGHBORS = 100           # Jumlah tetangga untuk Co-Like KNN
USER_CF_NEIGHBORS = 80            # Jumlah tetangga untuk User-Based CF

BEST_WEIGHTS = {
    "popularity": 0.0,
    "content":    0.2,
    "colike":     0.2,
    "author":     0.2,
    "usercf":     0.4,
}
```

---

## 🔄 Pipeline Data

```
Raw Books.csv + Ratings.csv + Users.csv
        │
        ▼
   Manual fixes (3 ISBN bermasalah)
        │
        ▼
   Normalisasi teks (judul, penulis, penerbit)
        │
        ▼
   Dedup ISBN → Buku Kanonik (book_id)
        │
        ▼
   Filter rating eksplisit (1–10)
        │
        ▼
   Warm filter (≥10 rating/user & ≥10 rating/buku)
        │
        ▼
   Train / Validation / Test Split
   (2 item relevan per user: 1 val, 1 test)
        │
        ▼
   Build 5 signal models
        │
        ▼
   Grid search bobot pada Validation
        │
        ▼
   Evaluasi final pada Test Set (250 users)
```

---

## 🎨 Desain UI

Dashboard menggunakan desain tema gelap premium dengan:

- **Font:** Syne (heading) + DM Sans (body) via Google Fonts
- **Palet:** Deep purple/violet gradient dengan aksen cyan dan hijau
- **Komponen:** Book card dengan cover image, rank badge, score badge
- **Chart:** Plotly dengan tema transparan yang terintegrasi dengan background
- **Sidebar:** Glassmorphism panel dengan logo BookMind

---

## 📝 Catatan Teknis

- **Caching:** `@st.cache_data` untuk data prep, `@st.cache_resource` untuk model (satu kali build per sesi)
- **Cover Buku:** Diambil dari URL `Image-URL-M` di dataset; fallback placeholder jika URL tidak valid
- **Rating 0:** Diperlakukan sebagai interaksi implisit, tidak digunakan dalam training sinyal eksplisit
- **Candidate Pool:** Setiap sinyal menghasilkan kandidat terbatas (400) sebelum scoring hybrid untuk efisiensi

---

## 📄 Lisensi

Dataset Book-Crossing: © Cai-Nicolas Ziegler, University of Freiburg.  
Kode dashboard: MIT License.
