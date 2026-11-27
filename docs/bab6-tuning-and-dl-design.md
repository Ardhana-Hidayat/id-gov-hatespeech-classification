# BAB 6 (LANJUTAN) — TUNING MODEL & DESAIN DEEP LEARNING (MINGGU 13)

**Proyek:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning  
**Mata Kuliah:** Data Science (Semester 5)  
**Tim Pengusul:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)  

Seluruh angka pada dokumen ini berasal dari eksekusi notebook:
- `notebooks/04_tuning_baseline_ml.ipynb` (PIC: Ardhana Syah Hidayat)
- `notebooks/04_deep_learning_design.ipynb` (PIC: Ardhana Syah Hidayat)
- `notebooks/04_tuning_advanced_ml.ipynb` (PIC: Muhammad Adam Al-Kidri — *Menunggu Track 2*)

> [!IMPORTANT]
> **Protokol Integritas Data:** Seluruh *hyperparameter tuning* hanya menggunakan *5-Fold Stratified Cross-Validation* pada `data/processed/train.csv` (9.105 baris) dan evaluasi pada `data/processed/val.csv` (1.951 baris). File `data/processed/test.csv` **tetap disegel** sampai Minggu 6.

---

## BAGIAN A — TUNING MODEL 1: BASELINE (TF-IDF + LOGISTIC REGRESSION)
*Penanggung Jawab (PIC): Ardhana Syah Hidayat (244311037)*

### A.1 Search Space & Best Parameters (`GridSearchCV`, 5-Fold Stratified CV)
Pencarian dilakukan dalam `Pipeline` scikit-learn agar `TfidfVectorizer` hanya di-*fit* pada bagian latih di setiap *fold* (*zero data leakage*):
* `tfidf__ngram_range`: `[(1, 1), (1, 2)]`
* `tfidf__max_features`: `[5000, 10000, 15000]`
* `clf__C`: `[0.5, 1.0, 2.0, 5.0]`
* Konfigurasi tetap: `tfidf__min_df=2`, `tfidf__sublinear_tf=True`, `clf__class_weight='balanced'`, `clf__solver='lbfgs'`

**Hasil Pencarian (24 kandidat $\times$ 5 fold = 120 *fits*):**
* **Best CV Macro F1:** `0,8323`
* **Best Parameters:** `{'clf__C': 2.0, 'tfidf__max_features': 10000, 'tfidf__ngram_range': (1, 2)}`

### A.2 Perbandingan Performa Sebelum vs Sesudah Tuning (Validation Set, $N = 1.951$)

| Metrik Evaluasi | Sebelum Tuning (Minggu 3) | Sesudah Tuning (Minggu 13) | Selisih ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | 0,8414 | **0,8428** | **+0,0014** |
| **Precision Kelas 1 (HS)** | **0,8167** | 0,8119 | -0,0048 |
| **Recall Kelas 1 (HS)** | 0,8167 | **0,8277** | **+0,0110** |
| **ROC-AUC** | 0,9168 | **0,9207** | **+0,0039** |
| **PR-AUC** | 0,8794 | **0,8842** | **+0,0048** |

*Catatan Generalisasi:* `Train Macro F1 = 0,9301` vs `Validation Macro F1 = 0,8428`.

### A.3 Optimasi Decision Threshold ($\tau$)
Untuk memenuhi target etika SDG 16 (**Precision Kelas 1 $\ge 85\%$** guna meminimalkan *False Positive* pada kritik wajar):
* Pada ambang bawaan $\tau = 0,50$: Macro F1 = `0,8428`, Precision HS = `0,8119`, Recall HS = `0,8277`.
* Pada ambang $\tau = 0,60$: Precision HS melonjak menjadi **`0,8693`** (melampaui target $\ge 85\%$), dengan Recall HS = `0,7427` dan Macro F1 = `0,8365`.

---

## BAGIAN B — BAB 6.2.4–6.2.5: TUNING ADVANCED (XGBOOST) & KONTROL OVERFITTING
*Penanggung Jawab (PIC): Muhammad Adam Al-Kidri (244311050)*

*(Dialokasikan untuk diisi setelah eksekusi `notebooks/04_tuning_advanced_ml.ipynb`)*
* [ ] **Bab 6.2.4:** *Search space*, *best parameters*, dan skor sebelum vs sesudah tuning (`RandomizedSearchCV` / SVD atau TF-IDF *sparse*).
* [ ] **Bab 6.2.5:** Kontrol *overfitting* (`early_stopping_rounds`, regularisasi $L_1/L_2$, `subsample`, `colsample_bytree`, serta kurva *Train vs Validation*).

---

## BAGIAN C — BAB 6.3: DESAIN MODEL 3 — DEEP LEARNING (BI-LSTM)
*Penanggung Jawab (PIC): Ardhana Syah Hidayat (244311037)*

### 6.3.1 Pemilihan Jenis Deep Learning & Alasan
- **Jenis DL Terpilih:** **Recurrent Neural Network — 2-Layer Bidirectional LSTM (Bi-LSTM)** *(sesuai alternatif resmi pada proposal `docs/pengajuan-ide.md` dan `README.md`; opsi Transfer Learning IndoBERT dicadangkan untuk eksekusi GPU Cloud).*
- **Alasan Pemilihan Berdasarkan Karakteristik Data:**
  1. **Urutan Kata & Negasi:** Berbeda dari TF-IDF (*bag-of-words*), Bi-LSTM memproses sekuens token berurutan sehingga membedakan kalimat seperti *"tidak pantas menghina"* vs *"pantas dihina"*.
  2. **Konteks Dua Arah:** Pembacaan kiri-ke-kanan ($\rightarrow$) dan kanan-ke-kiri ($\leftarrow$) memperkaya representasi kata dengan konteks sebelum dan sesudahnya.
  3. **Kelayakan Komputasi:** Arsitektur ±2 juta parameter berjalan cepat dan stabil di CPU lokal tanpa memerlukan unduhan bobot eksternal berukuran besar.

### 6.3.2 Desain Arsitektur, Inductive Bias, & `model.summary()`

**Struktur Layer (`BiLSTMClassifier`):**
1. **Input:** Sekuens indeks integer `MAX_LEN = 80`.
2. **Embedding Layer:** `nn.Embedding(10000, 128, padding_idx=0)` + `Dropout(0.3)`.
3. **2-Layer Bidirectional LSTM:** `hidden_dim=128` per arah ($2 \times 128 = 256$ fitur kontekstual per langkah waktu), `dropout=0.3` *(memenuhi syarat minimum $\ge 1$ layer RNN)*.
4. **Dual Global Pooling:** `MeanPool(256)` + `MaxPool(256)` $\rightarrow$ vektor kalimat **512 dimensi**.
5. **Classification Head (MLP):** `Linear(512 -> 128)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Linear(128 -> 2)`.

**Ringkasan Parameter (`model.summary()`):**
* **Total Parameters:** `2,005,378`
* **Trainable Parameters:** `2,005,378`

**Inductive Bias Arsitektur:**
1. *Sequential Order & Gated Recurrence:* Urutan token menentukan makna kalimat dan gerbang memori menyaring dependensi jangka panjang-pendek.
2. *Bidirectional Symmetry:* Makna kata ditentukan oleh konteks kiri dan kanan secara bersamaan.
3. *Translation Invariance via Pooling:* Sinyal frasa ujaran kebencian tetap terdeteksi terlepas dari posisinya di awal, tengah, atau akhir cuitan.

### 6.3.3 Preprocessing Khusus Deep Learning
* **Varian Teks:** `text_clean_contextual` (kata negasi *tidak, bukan, jangan* tetap utuh).
* **Vocabulary:** 10.000 token teratas dibangun **hanya dari Training Set** (`<PAD>=0`, `<UNK>=1`).
* **Input Shape & Padding:** `MAX_LEN = 80` (panjang maksimum korpus adalah 52 kata sehingga tidak memotong kalimat), *batch size* `64` $\rightarrow$ tensor `(64, 80)`.

### 6.3.5 Hasil Training Awal (*Sanity Check* 3 Epoch)
* **Optimizer:** `Adam(lr=1e-3, weight_decay=1e-5)` dengan *gradient clipping* (`max_norm=1.0`) dan `CrossEntropyLoss`.

| Epoch | Train Loss | Validation Loss | Val Macro F1 | Val ROC-AUC | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0,5944 | 0,4742 | 0,7675 | 0,8516 | Loss mulai turun |
| **2** | 0,4540 | 0,4028 | 0,8152 | 0,8947 | Konvergensi membaik pesat |
| **3** | **0,3729** | **0,3923** | **0,8287** | **0,9083** | **Loss turun konsisten, bebas error** |

*Kesimpulan Sanity Check:* Pipeline Deep Learning terbukti berjalan *end-to-end* tanpa error, *Train Loss* turun dari `0,5944` ke `0,3729`, dan *Validation Loss* turun dari `0,4742` ke `0,3923` hanya dalam 3 *epoch* awal.
