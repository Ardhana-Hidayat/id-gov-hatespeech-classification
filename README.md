# Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning

![SDG 16](https://img.shields.io/badge/UN_SDG-16.10_Peace,_Justice_&_Strong_Institutions-00689D?style=for-the-badge)
![Task](https://img.shields.io/badge/Task-Indonesian_NLP_Text_Classification-blue?style=for-the-badge)
![Status](https://img.shields.io/badge/Checkpoint_1-Learning_Problem-success?style=for-the-badge)

Repository ini berisi dokumentasi, alur prapemrosesan bahasa alami (*Natural Language Processing*), dan eksperimen pemodelan untuk mengklasifikasikan ujaran kebencian (*hate speech*) pada diskursus informasi pemerintahan Indonesia di media sosial. Proyek ini dikembangkan guna mendukung ruang demokrasi digital yang sehat dan inklusif sesuai target **Sustainable Development Goals (SDG) 16**.

---

## Identitas Pengusul
* **Nama:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)
* **Mata Kuliah:** Data Science
* **Dosen Pengampu:** Gus Nanang Syaifuddin, S.Kom., M.Kom.

---

## Dokumen Deliverables (Minggu 1 — Checkpoint 1)
Sesuai dengan ketentuan **Minggu 1: [L] Learning Problem**, dokumen pengajuan ide dan proposal awal dapat diakses langsung melalui tautan berikut:
1. 📄 **[Lampiran B: Formulir Pengajuan Ide](./docs/pengajuan-ide.md)**
2. 📑 **[Proposal Singkat (Bab 1 – Bab 3 Versi Awal)](./docs/learning-problem.md)**

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
├── data/                              # Penyimpanan dataset lokal (diabaikan oleh Git)
├── docs/                              # Dokumentasi laporan dan pengajuan ide
│   ├── lampiran_b_formulir_ide.md     # Deliverable 1: Formulir Lampiran B (Checkpoint 1)
│   └── proposal_bab1_bab3.md          # Deliverable 2: Proposal Singkat Bab 1 - Bab 3
├── notebooks/                         # Jupyter / Google Colab Notebooks eksperimen
│   ├── 01_eda_and_text_cleaning.ipynb # Eksplorasi distribusi teks & normalisasi kamus alay
│   ├── 02_baseline_logreg_tfidf.ipynb # Pelatihan model Baseline (TF-IDF + Logistic Regression)
│   ├── 03_advanced_xgboost.ipynb      # Pelatihan model Advanced (Feature Engineering + XGBoost)
│   └── 04_dl_indobert_finetuning.ipynb# Pelatihan model Deep Learning (Fine-Tuning IndoBERT)
├── src/                               # Modul fungsi pembersihan teks dan evaluasi metrik
├── .gitignore                         # Konfigurasi pengabaian file dataset dan bobot model besar
├── requirements.txt                   # Daftar pustaka Python (scikit-learn, xgboost, transformers, torch)
└── README.md                          # Dokumentasi utama proyek