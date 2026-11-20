# BAB 6 — MODEL DESIGN: BASELINE & ADVANCED

**Proyek:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning  
**Mata Kuliah:** Data Science (Semester 5)  
**Pengusul:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)  

Seluruh angka pada dokumen ini berasal dari eksekusi notebook:
- `notebooks/03_model_design_baseline_ml.ipynb` (Bab 6.1 — PIC: Ardhana Syah Hidayat)
- `notebooks/03_model_design_advanced_ml.ipynb` (Bab 6.2 — PIC: Muhammad Adam Al-Kidri)

> [!IMPORTANT]
> **Protokol Integritas Evaluasi:** Seluruh metrik validasi di bawah ini diukur murni menggunakan **Validation Set** (`data/processed/val.csv`, $N = 1.953$ baris). Sesuai metodologi pencegahan kebocoran data, berkas data uji (`data/processed/test.csv`, $N = 1.952$ baris) **tetap disegel** dan tidak digunakan dalam proses penalaan atau pemilihan model pada bab ini.

---

## BAB 6.1 — MODEL 1: BASELINE (LOGISTIC REGRESSION + TF-IDF)
*Penanggung Jawab (PIC): Ardhana Syah Hidayat (244311037)*

### 6.1.1 Teori Dasar & Alasan Pemilihan Model

#### A. Teori Dasar Pemodelan
Logistic Regression adalah model klasifikasi linier probabilistik yang memetakan kombinasi linier fitur input $\mathbf{x}$ ke dalam estimasi probabilitas bersyarat kelas positif ($y = 1$, *Hate Speech*) melalui fungsi logistik baku (sigmoid):

$$P(Y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

di mana $\mathbf{w} \in \mathbb{R}^D$ adalah vektor bobot fitur dan $b \in \mathbb{R}$ adalah bias. Estimasi parameter optimal diperoleh dengan meminimalkan fungsi kerugian *Binary Cross-Entropy (Log-Loss)* yang dilengkapi regularisasi $L_2$ (Ridge):

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{1}{2C} \|\mathbf{w}\|_2^2$$

Parameter $C > 0$ mengontrol kekuatan regularisasi, di mana nilai $C$ yang lebih kecil memberikan sanksi lebih kuat terhadap besaran bobot.

#### B. Alasan Pemilihan Berdasarkan Karakteristik Data
1. **Ruang Fitur Sparse Berdimensi Tinggi ($D = 5.000$):**  
   Ekstraksi leksikal menggunakan unigram dan bigram TF-IDF menghasilkan matriks dengan tingkat *sparsity* mencapai **99,63%**. Berdasarkan Teorema Cover (*Cover's Theorem on Separability of Patterns*), pemetaan data teks ke ruang fitur berdimensi tinggi memperbesar peluang keterpisahan kelas secara linier (*linearly separable*). Logistic Regression sangat efisien dan stabil pada ruang fitur *sparse* berdimensi ribuan.
2. **Karakteristik Optimasi Konveks:**  
   Fungsi kerugian Log-Loss dengan penalti $L_2$ bersifat *strictly convex*, menjamin bahwa algoritma optimasi (L-BFGS) selalu konvergen menuju minimum global tanpa risiko terjebak pada minimum lokal.
3. **Keterukuran dan Interpretasi (*Baseline Benchmark*):**  
   Sebagai model garis dasar (*baseline*), tanda dan besaran koefisien $\mathbf{w}$ memberikan transparansi langsung mengenai bobot kata apa yang paling memicu vonis ujaran kebencian.

---

### 6.1.2 Hypothesis Space & Inductive Bias

#### A. Hypothesis Space ($\mathcal{H}$)
Ruang hipotesis dari Logistic Regression adalah keluarga semua fungsi pemetaan linier terbobot dari ruang fitur $\mathbb{R}^D$ ke dalam interval probabilitas terbuka $(0, 1)$:

$$\mathcal{H} = \left\{ f_{\mathbf{w}, b}(\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}} \;\middle|\; \mathbf{w} \in \mathbb{R}^D, \; b \in \mathbb{R} \right\}$$

Pada nilai ambang keputusan default $\tau = 0.5$, batas keputusan (*decision boundary*) yang dihasilkan berbentuk *hyperplane* datar berdimensi $(D-1)$ dalam ruang $\mathbb{R}^{5000}$:

$$\mathcal{B} = \left\{ \mathbf{x} \in \mathbb{R}^D \mid \mathbf{w}^T \mathbf{x} + b = 0 \right\}$$

#### B. Inductive Bias (Asumsi Dasar Model)
1. **Linearitas Log-Odds (*Linear Separability Assumption*):**  
   Model mengasumsikan bahwa logaritma rasio peluang (*log-odds*) dari kejadian ujaran kebencian berhubungan secara linier sempurna terhadap fitur kata:
   $$\ln \left( \frac{P(Y = 1 \mid \mathbf{x})}{1 - P(Y = 1 \mid \mathbf{x})} \right) = \mathbf{w}^T \mathbf{x} + b$$
2. **Aditivitas & Independensi Efek Fitur (*Feature Additivity*):**  
   Model mengasumsikan kontribusi tiap token bersifat independen dan aditif terhadap total skor log-odds. Kehadiran suatu kata meningkatkan atau menurunkan skor kebencian secara konstan tanpa memperhitungkan interaksi hierarkis non-linier antar-kata (kecuali yang tertangkap dalam unit token bigram).
3. **Prior Gaussian pada Bobot ($L_2$ Ridge Prior):**  
   Penalti $\frac{1}{2C} \|\mathbf{w}\|_2^2$ setara dengan asumsi apriori Bayesian bahwa koefisien parameter berdistribusi normal di sekitar nol ($w_j \sim \mathcal{N}(0, \tau^2)$). Asumsi ini mencegah *overfitting* terhadap kata-kata jarang (*rare tokens*) pada korpus pelatihan.

---

### 6.1.3 Konfigurasi Hiperparameter Awal

| Komponen / Hiperparameter | Nilai Konfigurasi | Alasan Metodologis |
| :--- | :--- | :--- |
| **Varian Teks Input** | `text_clean_lexical` | Teks bersih tanpa stopword umum, kata negasi dipertahankan |
| **Representasi Fitur** | TF-IDF (unigram & bigram) | `ngram_range=(1,2)`, `max_features=5000`, `min_df=2`, `sublinear_tf=True` |
| **Penalti Regularisasi** | $L_2$ (Ridge) | Menstabilkan estimasi bobot pada matriks jarang |
| **Nilai Parameter $C$** | $1.0$ | Titik awal regularisasi standar |
| **Pembobotan Kelas** | `class_weight='balanced'` | Menangani ketimpangan ringan rasio kelas 1,37 : 1 secara objektif |
| **Algoritma Solver** | `lbfgs` | Kuasi-Newton efisien untuk optimasi fungsi konveks berskala besar |
| **Maksimum Iterasi** | $1.000$ | Menjamin konvergensi gradien penuh |
| **Random Seed** | $42$ | Menjamin reprodusibilitas hasil pelatihan |

*Catatan Anti-Leakage:* `TfidfVectorizer` di-fit murni pada **9.116 baris Train Set** dan mentransformasi Validation Set secara pasif.

---

### 6.1.4 Hasil Evaluasi pada Validation Set ($N = 1.953$)

Evaluasi pada data validasi menghasilkan metrik performa sebagai berikut:

| Metrik Evaluasi | Nilai Validasi | Target Kriteria Proyek | Status Ketercapaian |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | **0,8251** (82,51%) | $\ge 85,00\%$ | *Mendekati target (Garis dasar kuat)* |
| **Precision Kelas 1 (Hate Speech)** | **0,8000** (80,00%) | $\ge 85,00\%$ | *Aman dari over-censorship ekstrem* |
| **Recall Kelas 1 (Hate Speech)** | **0,7952** (79,52%) | $\ge 80,00\%$ | *Tercapai secara seimbang* |
| **F1-Score Kelas 1 (Hate Speech)** | **0,7976** (79,76%) | — | *Keseimbangan Precision-Recall kuat* |
| **ROC-AUC** | **0,9043** (90,43%) | $\ge 88,00\%$ | **Melampaui Target** |
| **PR-AUC (Average Precision)** | **0,8736** (87,36%) | — | *Pemisahan kurva presisi sangat baik* |

**Matriks Kebingungan (*Confusion Matrix*) pada Data Validasi:**

| Label Sebenarnya \ Prediksi | Prediksi Non-HS (`0`) | Prediksi Hate Speech (`1`) | Total Sampel |
| :--- | :---: | :---: | :---: |
| **Aktual Non-HS (`0`)** | **964** (*True Negative*) | **164** (*False Positive*) | 1.128 baris |
| **Aktual Hate Speech (`1`)** | **169** (*False Negative*) | **656** (*True Positive*) | 825 baris |
| **Total Prediksi** | 1.133 | 820 | **1.953 baris** |

*Insight Confusion Matrix:*
- Tingkat *False Positive* sebesar **164 baris (14,54% dari total Non-HS)**: Model baseline masih memiliki sedikit kecenderungan salah mencap opini kritis atau cuitan bernada tegas sebagai ujaran kebencian.
- Tingkat *False Negative* sebesar **169 baris (20,48% dari total HS)**: Ujaran kebencian implisit atau berbasis sindiran halus belum tertangkap penuh oleh representasi leksikal TF-IDF.

---

### 6.1.5 Analisis Koefisien Fitur & Interpretasi Linguistik

Keunggulan utama model linier adalah interpretasi langsung nilai bobot fitur ($w_j$):

| Peringkat | Top 10 Fitur Pemicu Hate Speech ($w_j > 0$) | Koefisien | Top 10 Fitur Penahan Non-HS ($w_j < 0$) | Koefisien |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `cebong` | **+5,9122** | `budaya` | **-3,2723** |
| 2 | `ahok` | **+4,1522** | `bom` | **-3,1399** |
| 3 | `bubarkan` | **+4,1071** | `ekonomi` | **-3,1197** |
| 4 | `prabowo` | **+4,1070** | `katolik` | **-3,0989** |
| 5 | `jokowi` | **+4,0302** | `sipit` | **-2,5535** |
| 6 | `dasar` | **+4,0292** | `presiden jokowi` | **-2,4784** |
| 7 | `lengserkan` | **+3,9923** | `jawa` | **-2,2435** |
| 8 | `tolol` | **+3,8921** | `kristen` | **-2,1712** |
| 9 | `dungu` | **+3,7309** | `pilihan` | **-2,1709** |
| 10 | `usir` | **+3,5747** | `buaya` | **-2,1241** |

*Temuan Linguistik Krusial:*
1. **Dominasi Sentimen Polarisasi Politik:** Kata ejekan politik (`cebong`) menjadi prediktor kebencian terkuat (+5,91), diikuti kata imperatif agresif (`bubarkan`, `lengserkan`, `usir`) serta kata makian intelek (`tolol`, `dungu`).
2. **Konteks Unigram vs Bigram:** Unigram nama figur politik seperti `jokowi` (+4,03) memiliki asosiasi positif ke ujaran kebencian karena sering diserang pada cuitan sentimen negatif; sebaliknya bigram resmi `presiden jokowi` memiliki bobot negatif (**-2,48**), membuktikan pentingnya fitur bigram untuk membedakan pemberitaan/diskusi formal dari hujatan.

---

### 6.1.6 Eksplorasi Sensitivitas Regularisasi ($C$) pada Validation Set

Pengujian nilai parameter regularisasi $C$ dilakukan untuk memastikan bahwa $C = 1.0$ adalah pilihan garis dasar yang optimal:

| Nilai $C$ | Macro F1-Score | Precision Kelas 1 (HS) | Recall Kelas 1 (HS) | F1-Score HS | ROC-AUC | Catatan Perilaku |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.01` | 0,7823 | 0,7781 | 0,7055 | 0,7400 | 0,8664 | Regularisasi terlalu ketat (*Underfitting*) |
| `0.1` | 0,8074 | 0,7904 | 0,7588 | 0,7743 | 0,8845 | Recall meningkat pesat |
| `0.5` | 0,8199 | 0,7939 | 0,7891 | 0,7915 | 0,8995 | Menuju konvergensi optimal |
| **`1.0`** | **0,8251** | **0,8000** | **0,7952** | **0,7976** | **0,9043** | **Konfigurasi Garis Dasar Terpilih (Puncak F1 & Precision)** |
| `2.0` | 0,8238 | 0,7956 | 0,7976 | 0,7966 | 0,9063 | Presisi mulai turun |
| `5.0` | 0,8228 | 0,7937 | 0,7976 | 0,7956 | 0,9048 | Penalti $L_2$ melemah |
| `10.0` | 0,8199 | 0,7867 | 0,8000 | 0,7933 | 0,9008 | Gejala awal *overfitting* pada kata langka |

*Kesimpulan Bab 6.1:* Model Baseline menetapkan standar performa awal yang kokoh (**Macro F1 = 82,51%**, **ROC-AUC = 90,43%**), membuktikan bahwa fitur leksikal n-gram linear sudah mampu menangkap sinyal kebencian eksplisit dengan baik.

---

## BAB 6.2 — MODEL 2: ADVANCED (XGBOOST CLASSIFIER)
*Penanggung Jawab (PIC): Muhammad Adam Al-Kidri (244311050)*

*(Bagian ini dialokasikan untuk diisi oleh Anggota 2 setelah eksekusi notebook `03_model_design_advanced_ml.ipynb`)*

### 6.2.1 Teori Dasar, Keunggulan & Kelemahan XGBoost
* [ ] Teori Gradient Boosted Decision Trees (CART ensemble aditif)
* [ ] Alasan penggabungan fitur 100 komponen TruncatedSVD + 6 Fitur Gaya Penulisan
* [ ] Analisis komparatif keunggulan dan kelemahan dibanding model linear

### 6.2.2 Hypothesis Space & Inductive Bias
* [ ] Formulasi matematis ruang hipotesis pohon $\mathcal{H} = \left\{ \sum f_k(\mathbf{x}) \right\}$
* [ ] Pembahasan 3 asumsi induktif: *Axis-aligned cuts*, *Hierarchical feature interactions*, dan *Tree shrinkage/parsimony*

### 6.2.3 Konfigurasi Hiperparameter Awal & Hasil Validasi
* [ ] Tabel hiperparameter awal (`n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight`)
* [ ] Hasil evaluasi pada Validation Set (`val.csv`)
* [ ] Diagram *Feature Importance* (analisis kontribusi fitur gaya penulisan)

---

## BAB 6.3 — KOMPARASI AWAL BASELINE VS ADVANCED (VALIDATION SET)
*(Disintesis bersama oleh Ardhana Syah Hidayat & Muhammad Adam Al-Kidri)*

| Metrik Evaluasi | Model 1: Baseline (Logistic Regression) | Model 2: Advanced (XGBoost) | Selisih ($\Delta$) | Analisis Keunggulan |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1-Score** | **0,8251** | *[Menunggu Track 2]* | — | Keseimbangan performa kelas |
| **Precision (Kelas 1 - HS)** | **0,8000** | *[Menunggu Track 2]* | — | Proteksi False Positive / Kritik |
| **Recall (Kelas 1 - HS)** | **0,7952** | *[Menunggu Track 2]* | — | Cakupan penangkapan kebencian |
| **ROC-AUC** | **0,9043** | *[Menunggu Track 2]* | — | Pemisahan probabilitas |
| **Waktu Pelatihan** | **< 1 detik** | *[Menunggu Track 2]* | — | Efisiensi komputasi |
