# Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning

![SDG 16](https://img.shields.io/badge/UN_SDG-16.10_Peace,_Justice_&_Strong_Institutions-00689D?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Indonesian_NLP_Text_Classification-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Checkpoint_2-Data_Preparation-success?style=for-the-badge)

Repository ini berisi dokumentasi, alur prapemrosesan bahasa alami (*Natural Language Processing*), dan eksperimen pemodelan untuk mengklasifikasikan ujaran kebencian (*hate speech*) pada diskursus informasi pemerintahan Indonesia di media sosial. Proyek ini dikembangkan guna mendukung ruang demokrasi digital yang sehat dan inklusif sesuai target **Sustainable Development Goals (SDG) 16**.

---

## Identitas Pengusul
* **Nama:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)
* **Mata Kuliah:** Data Science
* **Dosen Pengampu:** Gus Nanang Syaifuddin, S.Kom., M.Kom.

---

## Dokumen Deliverables
Dokumen pengajuan ide, proposal awal, serta hasil *data understanding* dan *data preparation* dapat diakses langsung melalui tautan berikut:
1. 📄 **[Lampiran B: Formulir Pengajuan Ide](./docs/pengajuan-ide.md)**
2. 📑 **[Proposal Singkat (Bab 1 – Bab 3 Versi Awal)](./docs/learning-problem.md)**
3. 📊 **[Bab 4 – 5: Data Understanding & Data Preparation](./docs/bab4-bab5-data-preparation.md)**

---

## Ringkasan Learning Problem & Keterkaitan SDG

### 1. Kontribusi terhadap SDG 16 (Target 16.7 & 16.10)
Maraknya ujaran kebencian pada kolom diskusi kebijakan pemerintah mengaburkan kritik dan aspirasi warga negara yang konstruktif. Sistem klasifikasi NLP ini dirancang sebagai fondasi penyaringan otomatis untuk memisahkan ujaran kebencian dari kritik kebijakan yang sah, sehingga mendukung penyerapan aspirasi publik yang responsif sekaligus menjaga ruang informasi yang sehat.

### 2. Spesifikasi Learning Problem

| Komponen | Deskripsi Teknis |
| :--- | :--- |
| **Jenis Tugas ML** | *Supervised Learning — Binary Text Classification (NLP)* |
| **Target ($Y$)** | Kelas `1`: **Hate Speech** (Mengandung ujaran kebencian/hasutan pada wacana publik)<br>Kelas `0`: **Non-Hate Speech** (Informasi netral, diskusi umum, atau kritik kebijakan yang sah) |
| **Data ($X$)** | Teks unggahan media sosial berbahasa Indonesia (dinormalisasi dengan kamus *slang/alay* dan diekstraksi menjadi vektor fitur / token sekuens) |
| **Objective** | Memaksimalkan ketepatan deteksi ujaran kebencian sekaligus meminimalkan kesalahan pelabelan pada kritik masyarakat yang sah (*minimizing False Positives*) |
| **Metrik Evaluasi** | **Macro F1-Score** ($\ge 85\%$), **ROC-AUC** ($\ge 88\%$), serta **Precision** & **Recall** per kelas |

---

## Dataset
Proyek ini menggunakan dataset publik berbahasa Indonesia yang telah tervalidasi secara ilmiah:
* **Nama Dataset:** [Indonesian Abusive and Hate Speech Twitter Text](https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text) (Ibrohim & Budi, 2019).
* **Ukuran & Proporsi:** **13.169 baris teks** berbahasa Indonesia (~42,2% *Hate Speech* dan ~57,8% *Non-Hate Speech*), dilengkapi kamus normalisasi kata tidak baku (`new_kamusalay.csv`).
* **Strategi Pencegahan *Data Leakage*:** Pembagian data (*70% Train, 15% Validation, 15% Test*) dilakukan secara *stratified* sebelum proses ekstraksi fitur. Pembuatan kamus frekuensi kata (`TfidfVectorizer` / `Tokenizer`) hanya di-*fit* pada *Training Set*.

**Hasil data preparation** (detail di [Bab 4–5](./docs/bab4-bab5-data-preparation.md)):

| Split | Baris | Proporsi `HS=1` |
| :--- | ---: | ---: |
| Train | 9.105 | 42,26% |
| Validation | 1.951 | 42,23% |
| Test | 1.952 | 42,26% |

Dari 13.169 baris mentah: 28 baris berlabel konflik dan 126 duplikat teks dibuang, lalu 7 dokumen kosong setelah *cleaning* dibuang → **13.008 baris** final (`random_state=42`). Test set disegel sampai Minggu 6.

---

## Rencana Arsitektur Tiga Model

| Tingkatan Model | Arsitektur / Metode | Input Fitur | Tujuan Eksperimen |
| :--- | :--- | :--- | :--- |
| **1. Baseline** | **Logistic Regression** | Vektor **TF-IDF** (*Unigram & Bigram*, `max_features=5000`) | Menetapkan batas bawah (*benchmark*) performa klasifikasi linier dan menganalisis bobot kata kunci utama. |
| **2. Advanced** | **XGBoost Classifier** | Komponen **TruncatedSVD (TF-IDF)** + *Engineered Features* (rasio huruf kapital, panjang teks, kepadatan tanda baca/kata kasar) | Menguji kemampuan algoritma *gradient boosting* dalam menangkap interaksi non-linier antara gaya penulisan dan fitur leksikal. |
| **3. Deep Learning** | **IndoBERT (`indobert-base-p1`) / Bi-LSTM** | Sekuens *Token IDs* & *Attention Mask* | Memanfaatkan arsitektur *Transformer pre-trained* bahasa Indonesia untuk memahami konteks kalimat dua arah dan sarkasme/implisit. |

---

## Struktur Direktori Repository

```text
├── data/                              # Dataset lokal (diabaikan oleh Git)
│   ├── raw/                           # data.csv, new_kamusalay.csv, abusive.csv
│   └── processed/                     # train.csv, val.csv, test.csv, split_manifest.json
├── docs/                              # Dokumentasi laporan dan pengajuan ide
│   ├── pengajuan-ide.md               # Deliverable 1: Formulir Lampiran B
│   ├── learning-problem.md            # Deliverable 2: Proposal Bab 1 - Bab 3
│   └── bab4-bab5-data-preparation.md  # Deliverable 3: Data understanding & preparation
├── notebooks/                         # Jupyter Notebook eksperimen
│   ├── 01_learning_problem_eda.ipynb  # Bab 4: profil data, kondisi data, 5 visualisasi EDA
│   ├── 02_data_preparation.ipynb      # Bab 5: cleaning, feature engineering, split, transformasi
│   ├── 03_baseline_logreg_tfidf.ipynb # (Minggu 3) Baseline: TF-IDF + Logistic Regression
│   ├── 04_advanced_xgboost.ipynb      # (Minggu 4) Advanced: SVD + fitur linguistik + XGBoost
│   └── 05_dl_indobert_finetuning.ipynb# (Minggu 5) Deep Learning: Fine-tuning IndoBERT
├── src/
│   └── check_data_preparation.py      # Cek artefak split: baris, kebocoran teks, hash test set
├── .gitignore                         # Mengabaikan dataset, model weights, dan cache
├── requirements.txt                   # Daftar pustaka Python
└── README.md                          # Dokumentasi utama proyek
```
