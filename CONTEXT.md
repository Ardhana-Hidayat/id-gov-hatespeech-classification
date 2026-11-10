# PROJECT CONTEXT & METHODOLOGY SPECIFICATION (`CONTEXT.md`)

Dokumen ini merupakan acuan teknis dan metodologis utama (*Single Source of Truth*) untuk proyek **Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning**. Seluruh penulisan kode, *notebook* eksperimen, dan laporan wajib mematuhi batasan di dalam dokumen ini.

---

## 1. Identitas & Ruang Lingkup Proyek
* **Judul Proyek:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning
* **Domain Tugas:** *Natural Language Processing* (NLP) — *Supervised Binary Text Classification* (Bahasa Indonesia)
* **Keterkaitan SDG:** **SDG 16** (*Peace, Justice and Strong Institutions*) — **Target 16.10** (Akses informasi publik & perlindungan kebebasan mendasar) dan **Target 16.7** (Pengambilan keputusan yang responsif, inklusif, dan partisipatif).
* **Masalah Inti:** Memisahkan ujaran kebencian (*hate speech*) yang destruktif dari kritik kebijakan/opini publik yang sah pada wacana informasi pemerintahan di media sosial Indonesia.

---

## 2. Spesifikasi Dataset & Skema Data

### 2.1 Sumber Data
* **Dataset Utama:** *Indonesian Abusive and Hate Speech Twitter Text Dataset* (Ibrohim & Budi, 2019).
* **URL Kaggle:** `https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text`
* **File Pendukung:**
  1. `data.csv` (~13.169 baris teks beserta label biner & multi-label).
  2. `new_kamusalay.csv` (~15.167 pasangan kata tidak baku/*slang* $\rightarrow$ kata baku bahasa Indonesia).
  3. `abusive.csv` (Daftar leksikon kata kasar bahasa Indonesia).

### 2.2 Variabel Target & Fitur
* **Input Utama ($X_{raw}$):** Kolom `Tweet` (teks unggahan media sosial berbahasa Indonesia).
* **Target Utama ($Y$):** Kolom `HS` (*Binary Target*):
  * `1` = **Hate Speech** (Mengandung ujaran kebencian, sekitar ~42.2%).
  * `0` = **Non-Hate Speech** (Informasi netral, diskusi umum, atau kritik sah tanpa unsur kebencian, sekitar ~57.8%).
* **Kolom Konteks Tambahan (Hanya untuk EDA & Error Analysis, BUKAN Fitur Input $X$!):**
  * Kolom seperti `Abusive`, `HS_Individual`, `HS_Group`, `HS_Religion`, `HS_Race`, `HS_Physical`, `HS_Gender`, `HS_Other`, `HS_Weak`, `HS_Moderate`, `HS_Strong` **DILARANG** dimasukkan sebagai fitur prediktor ($X$) saat melatih model karena akan menyebabkan **Target Leakage** fatal. Kolom-kolom ini hanya boleh dipakai saat *Exploratory Data Analysis* (EDA) dan *Error Analysis* setelah model selesai memprediksi.

---

## 3. Aturan Ketat Pencegahan *Data Leakage* & Prapemrosesan

Setiap tahapan di dalam `notebooks/` wajib mengikuti urutan eksekusi berikut:

1. **Deduplikasi Awal:**
   * Hapus baris yang memiliki teks `Tweet` identik 100% (*exact duplicates*) sebelum pembagian data agar teks yang sama tidak berada di *Train* dan *Test* secara bersamaan.
2. **Ekstraksi Fitur Gaya Penulisan (*Pre-Cleaning Feature Engineering*):**
   * **WAJIB dilakukan sebelum teks diubah ke huruf kecil atau dibersihkan!**
   * Ekstrak fitur numerik berikut dari `Tweet` mentah:
     * `char_length`: Panjang karakter total.
     * `word_count`: Jumlah kata dalam cuitan.
     * `caps_ratio`: Rasio huruf kapital terhadap total huruf ($\frac{\text{jumlah huruf kapital}}{\text{char\_length} + 1}$) sebagai indikator teriakan/emosi tinggi.
     * `exclamation_count`: Jumlah tanda seru (`!`) dan tanda tanya (`?`).
     * `user_mention_count`: Frekuensi kemunculan token `USER` (jumlah akun yang di-*mention*).
     * `abusive_lexicon_count`: Jumlah kata dalam cuitan yang cocok dengan daftar kata di `abusive.csv`.
3. **Pembersihan Teks & Normalisasi (*Text Cleaning*):**
   * Buat dua versi teks bersih untuk mengakomodasi perbedaan kebutuhan model klasik vs *Transformer*:
     * `text_clean_contextual` (Untuk **Model 3 - IndoBERT**): Hapus token `USER`, `URL`, `RT`, kode *hex* karakter (`\xf0...`), dan tanda baca berlebih, lalu ubah ke huruf kecil dan normalisasi kata menggunakan `new_kamusalay.csv`. **JANGAN hapus *stopwords* (kata seperti "tidak", "bukan", "jangan")** agar konteks negasi tetap utuh.
     * `text_clean_lexical` (Untuk **Model 1 & 2 - TF-IDF**): Sama seperti di atas, namun dapat dilengkapi dengan penghapusan *stopwords* umum (kecuali kata negasi penting) untuk mengurangi *noise* dimensi matriks.
4. **Pembagian Data (*Stratified Splitting*):**
   * Rasio pembagian: **70% Train**, **15% Validation**, **15% Test** (`stratify=y`, `random_state=42`).
   * Seluruh transformasi statistik (`TfidfVectorizer`, `TruncatedSVD`, `StandardScaler`) **HANYA boleh di-`.fit()` pada *Train Set***, kemudian di-`.transform()` pada *Validation* dan *Test Set*.

---

## 4. Spesifikasi Arsitektur Tiga Model

| Model | Nama File Notebook | Fitur Input ($X$) | Algoritma & Konfigurasi Utama |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | `02_baseline_logreg_tfidf.ipynb` | Matriks jarang **TF-IDF** dari `text_clean_lexical` (`ngram_range=(1,2)`, `max_features=5000`) | `LogisticRegression(class_weight='balanced', random_state=42)` dengan pencarian parameter regularisasi `C` pada *Validation Set*. |
| **2. Advanced** | `03_advanced_xgboost.ipynb` | Gabungan **100 komponen TruncatedSVD (TF-IDF)** + **6 Fitur Rekayasa Gaya Penulisan** (`caps_ratio`, dll.) | `XGBClassifier(eval_metric='logloss', random_state=42)` dengan *early stopping* pada *Validation Set* serta analisis *Feature Importance*. |
| **3. Deep Learning** | `04_dl_indobert_finetuning.ipynb` | Tokenisasi `text_clean_contextual` (`input_ids` & `attention_mask`, `max_length=128`) | *Fine-tuning* **IndoBERT** (`indobenchmark/indobert-base-p1`), `AdamW` (`lr=2e-5`), `batch_size=16/32`, `epochs=3-4`. |

---

## 5. Protokol Evaluasi & Target Keberhasilan

* **Metrik Utama:**
  1. **Macro F1-Score** (Target $\ge 85\%$): Mengukur keseimbangan performa pada kelas `0` dan `1` tanpa terdistorsi oleh ketimpangan jumlah sampel.
  2. **Precision Kelas `1` (*Hate Speech*)** (Target $\ge 85\%$): Krusial secara etika untuk meminimalkan **False Positive** (mencegah kritik warga yang sah terhadap pemerintah disensor/dilabeli sebagai ujaran kebencian).
  3. **Recall Kelas `1` (*Hate Speech*)**: Mengukur kemampuan sistem menangkap ujaran kebencian yang sebenarnya.
  4. **ROC-AUC & PR-AUC**: Mengukur kualitas pemisahan probabilitas pada berbagai *threshold*.
* **Analisis Wajib Pasca-Pemodelan (*Error Analysis*):**
  * Setiap *notebook* pemodelan wajib menampilkan **Confusion Matrix** dan mengambil sampel minimal 5 baris **False Positive** serta 5 baris **False Negative** dari *Validation/Test Set* untuk dianalisis penyebab kesalahannya secara linguistik (misal: penggunaan sarkasme, kata kiasan politik, atau singkatan yang belum masuk kamus).

---

## 6. Standar Penulisan Kode & Reprodusibilitas
* **Random Seed:** Gunakan `SEED = 42` pada seluruh operasi acak (`numpy`, `scikit-learn`, `torch`).
* **Penyimpanan Artefak:**
  * Data hasil prapemrosesan disimpan di `data/processed/` (`train.csv`, `val.csv`, `test.csv`).
  * Tabel ringkasan metrik evaluasi setiap model disimpan ke `results/metrics_summary.csv` agar mudah digabungkan menjadi grafik perbandingan akhir.