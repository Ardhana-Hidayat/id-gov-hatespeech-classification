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

### B.1 Eksperimen Representasi Fitur: Mengatasi Information Bottleneck
Pada evaluasi awal Minggu 3, XGBoost hanya mencapai **Macro F1 = 82,22%** dan tertinggal dari Baseline (**84,14%**). Analisis menunjukkan bahwa 100 komponen TruncatedSVD hanya merangkum **23,45%** varians TF-IDF (*information bottleneck*).

Untuk menguji solusi representasi, dilakukan eksperimen komparasi pada dua skenario:
1. **Skenario A (Variasi Dimensi Laten SVD):** Menguji `n_components = 100, 200, 300` yang difusikan dengan 6 fitur gaya penulisan ternormalisasi (`StandardScaler`).
2. **Skenario B (Sparse TF-IDF Direct Fusion):** Menggabungkan langsung matriks jarang TF-IDF 5.000 fitur dengan 6 fitur gaya penulisan (`scipy.sparse.hstack`, total 5.006 fitur) tanpa reduksi dimensi SVD.

| Skenario Representasi | Dimensi Fitur | Varians TF-IDF Terjelaskan | Macro F1 Validasi | Precision HS | Recall HS | ROC-AUC | Waktu Pelatihan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Skenario A: SVD-100 + Gaya (Minggu 3)** | 106 | 23,45% | 0,8222 | 0,8055 | 0,7791 | 0,9034 | 1,43 detik |
| **Skenario A: SVD-200 + Gaya (Terpilih)** | **206** | **34,12%** | **0,8234** | 0,8054 | **0,7828** | **0,9058** | 2,75 detik |
| **Skenario A: SVD-300 + Gaya** | 306 | 41,61% | 0,8149 | 0,7989 | 0,7718 | 0,9012 | 4,99 detik |
| **Skenario B: Sparse TF-IDF + Gaya** | 5.006 | 100,00% (Utuh) | 0,7989 | 0,7842 | 0,7427 | 0,8854 | 1,38 detik |

*Temuan Krusial Representasi:*
- **SVD-200 memberikan titik optimal:** Peningkatan ke 200 komponen menaikkan varians terjelaskan dari 23,45% ke **34,12%**, yang langsung meningkatkan Macro F1 ke **0,8234** dan Recall HS ke **0,7828**.
- **SVD-300 mengalami degradasi:** Retensi varians 41,61% justru menurunkan Macro F1 ke 0,8149 karena komponen laten tingkat tinggi membawa *noise* frekuensi rendah yang memicu *overfitting* pada split awal pohon.
- **Sparse TF-IDF gagal mengungguli SVD:** Pada matriks dengan *sparsity* >99,8% dan 5.000 kolom, pohon keputusan (*axis-aligned splits*) mengalami fragmentasi sinyal leksikal. Sinyal ujaran kebencian tersebar ke ribuan kolom jarang, sehingga boosting tanpa reduksi dimensi menghasilkan performa terendah (0,7989).
- **Keputusan:** Representasi **SVD-200 + 6 Fitur Gaya Penulisan (206 fitur padat)** ditetapkan sebagai ruang fitur resmi untuk tahap *Hyperparameter Tuning*.

---

### B.2 Hyperparameter Tuning (`RandomizedSearchCV`, 5-Fold Stratified CV)
Pencarian hiperparameter dilakukan secara sistematis menggunakan `RandomizedSearchCV` dengan **5-Fold Stratified Cross-Validation** hanya pada Training Set (`train.csv`, 9.105 baris).

**Search Space (8 Hiperparameter):**
* `n_estimators`: `[100, 200, 300, 500]`
* `max_depth`: `[3, 4, 6, 8]`
* `learning_rate`: `[0.03, 0.05, 0.1, 0.15]`
* `min_child_weight`: `[1, 3, 5]`
* `subsample`: `[0.7, 0.8, 0.9, 1.0]`
* `colsample_bytree`: `[0.6, 0.8, 1.0]`
* `reg_alpha` ($L_1$): `[0.0, 0.1, 1.0]`
* `reg_lambda` ($L_2$): `[1.0, 2.0, 5.0]`

**Hasil Pencarian (25 kandidat $\times$ 5 fold = 125 *fits*):**
* **Best CV Macro F1:** `0,8319` (meningkat signifikan dari estimasi CV awal ~0,815)
* **Best Hyperparameters Terpilih:**
  * `n_estimators`: `500` (dengan *Early Stopping* dinamis)
  * `learning_rate`: `0.03` (*shrinkage* konservatif untuk konvergensi halus)
  * `max_depth`: `8` (menangkap interaksi hierarkis non-linear hingga orde 8)
  * `min_child_weight`: `1` (memungkinkan partisi daun sensitif terhadap sinyal lokal)
  * `subsample`: `0.9` (stokastisitas baris 90% untuk meredam varians)
  * `colsample_bytree`: `0.8` (stokastisitas fitur 80% per pohon)
  * `reg_alpha` ($L_1$): `0.0`
  * `reg_lambda` ($L_2$): `2.0` (penalti kuadratik bobot daun diperkuat dari default 1.0)

---

### B.3 Kontrol Overfitting (Bab 6.2.5)
Untuk menahan laju *overfitting* yang umum terjadi pada model pohon berkapasitas tinggi (`max_depth=8`), diterapkan mekanisme kontrol bertingkat:
1. **Early Stopping Dinamis (`early_stopping_rounds=20`):**
   Pelatihan dievaluasi menggunakan fungsi kerugian `logloss` pada Validation Set. Pelatihan secara otomatis berhenti pada **iterasi ke-327** (dari total 500 pohon yang dialokasikan), mencegah 173 pohon tambahan yang berpotensi menghafal kebisingan data latih.
2. **Analisis Kurva Belajar (*Learning Curve*):**
   Kurva *Train Log-Loss vs Validation Log-Loss* menunjukkan bahwa konvergensi validasi mencapai titik minimum global pada iterasi 327 sebelum mulai mendatar (*plateau*).
3. **Analisis Gap Generalisasi:**
   * *Train Macro F1:* `0,9978`
   * *Validation Macro F1:* `0,8324`
   * Penalti regularisasi $L_2$ (`reg_lambda=2.0`), subsampling baris (`0.9`), dan kolom (`0.8`) berhasil menjaga kestabilan batas keputusan pada data validasi yang belum pernah dilihat.

---

### B.4 Perbandingan Performa Sebelum vs Sesudah Tuning (Validation Set, $N = 1.951$)

| Metrik Evaluasi | Sebelum Tuning (Minggu 3) | Sesudah Tuning (Minggu 13) | Selisih ($\Delta$) | Analisis Peningkatan |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1-Score** | 0,8222 | **0,8324** | **+0,0102 (+1,02%)** | Peningkatan signifikan melampaui 83% |
| **Precision Kelas 1 (HS)** | 0,8055 | **0,8229** | **+0,0174 (+1,74%)** | Menekan kesalahan False Positive |
| **Recall Kelas 1 (HS)** | 0,7791 | **0,7840** | **+0,0049 (+0,49%)** | Tangkapan ujaran kebencian meningkat |
| **F1-Score Kelas 1 (HS)** | 0,7921 | **0,8030** | **+0,0109 (+1,09%)** | Keseimbangan presisi-recall menguat |
| **ROC-AUC** | 0,9034 | **0,9106** | **+0,0072 (+0,72%)** | Pemisahan probabilitas kelas makin tajam |
| **PR-AUC** | 0,8806 | **0,8900** | **+0,0094 (+0,94%)** | Area presisi-recall melampaui Baseline |

**Matriks Kebingungan (*Confusion Matrix*) Sebelum vs Sesudah Tuning:**
- **Sebelum Tuning (Minggu 3):** TN = 972, FP = 155, FN = 182, TP = 642
- **Sesudah Tuning (Minggu 13):** **TN = 988 (+16)**, **FP = 139 (-16)**, **FN = 178 (-4)**, **TP = 646 (+4)**
- *Dampak Nyata:* Penurunan False Positive sebesar 16 sampel membuktikan model lebih aman terhadap kritik wajar (selaras SDG 16), sekaligus mampu mendeteksi 4 sampel ujaran kebencian tambahan yang sebelumnya terlewat.

---

### B.5 Optimasi Decision Threshold ($\tau \in [0.30, 0.70]$)
Untuk memenuhi kriteria etis SDG 16 (Target 16.10: Precision Kelas 1 $\ge 85\%$ untuk meminimalisasi sensor kritik kebijakan wajar):
* **Pada ambang default $\tau = 0,50$:** Macro F1 = `0,8324`, Precision HS = `0,8229`, Recall HS = `0,7840`.
* **Pada ambang optimal $\tau = 0,55$:** Precision HS berhasil melampaui target etika menjadi **`0,8525` (85,25%)**, dengan Macro F1 yang tetap kompetitif di `0,8294` dan Recall HS = `0,7403`.
* **Pada ambang $\tau = 0,60$:** Precision HS meningkat menjadi `0,8794` dengan Macro F1 = `0,8165`.

---

### B.6 Analisis Feature Importance Model Tuned
Pada konfigurasi 206 fitur padat:
* **Komponen SVD (200 fitur):** Menyumbang **92,15%** total *gain importance*, dipimpin oleh `SVD_008`, `SVD_018`, `SVD_005`, dan komponen baru `SVD_142`.
* **Fitur Gaya Penulisan (6 fitur):** Menyumbang **7,85%** total *gain importance*. Fitur `user_mention_count` (peringkat 6), `abusive_lexicon_count` (peringkat 8), dan `char_length` (peringkat 11) tetap berada di jajaran 15 fitur paling berpengaruh dalam memandu partisi pohon.

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
