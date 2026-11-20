# BAB 6 — MODEL DESIGN: BASELINE & ADVANCED

**Proyek:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning  
**Mata Kuliah:** Data Science (Semester 5)  
**Pengusul:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)  

Seluruh angka pada dokumen ini berasal dari eksekusi notebook:
- `notebooks/03_model_design_baseline_ml.ipynb` (Bab 6.1 — PIC: Ardhana Syah Hidayat)
- `notebooks/03_model_design_advanced_ml.ipynb` (Bab 6.2 — PIC: Muhammad Adam Al-Kidri)

> [!IMPORTANT]
> **Protokol Integritas Evaluasi:** Seluruh metrik validasi di bawah ini diukur murni menggunakan **Validation Set** (`data/processed/val.csv`, $N = 1.951$ baris) dari split final Bab 5 (**9.105 Train / 1.951 Validation / 1.952 Test**). Sesuai metodologi pencegahan kebocoran data, berkas data uji (`data/processed/test.csv`, $N = 1.952$ baris) **tetap disegel** dan tidak digunakan dalam proses penalaan atau pemilihan model pada bab ini.

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
   Ekstraksi leksikal menggunakan unigram dan bigram TF-IDF menghasilkan matriks dengan tingkat *sparsity* mencapai **99,81%**. Berdasarkan Teorema Cover (*Cover's Theorem on Separability of Patterns*), pemetaan data teks ke ruang fitur berdimensi tinggi memperbesar peluang keterpisahan kelas secara linier (*linearly separable*). Logistic Regression sangat efisien dan stabil pada ruang fitur *sparse* berdimensi ribuan.
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

*Catatan Anti-Leakage:* `TfidfVectorizer` di-fit murni pada **9.105 baris Train Set** dan mentransformasi Validation Set secara pasif.

---

### 6.1.4 Hasil Evaluasi pada Validation Set ($N = 1.951$)

Evaluasi pada data validasi menghasilkan metrik performa sebagai berikut:

| Metrik Evaluasi | Nilai Validasi | Target Kriteria Proyek | Status Ketercapaian |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | **0,8414** (84,14%) | $\ge 85,00\%$ | *Mendekati target (Garis dasar kuat)* |
| **Precision Kelas 1 (Hate Speech)** | **0,8167** (81,67%) | $\ge 85,00\%$ | *Belum mencapai target, namun FP relatif terkendali* |
| **Recall Kelas 1 (Hate Speech)** | **0,8167** (81,67%) | $\ge 80,00\%$ | *Tercapai* |
| **F1-Score Kelas 1 (Hate Speech)** | **0,8167** (81,67%) | — | *Keseimbangan Precision-Recall kuat* |
| **ROC-AUC** | **0,9168** (91,68%) | $\ge 88,00\%$ | **Melampaui Target** |
| **PR-AUC (Average Precision)** | **0,8794** (87,94%) | — | *Pemisahan kurva presisi sangat baik* |

**Matriks Kebingungan (*Confusion Matrix*) pada Data Validasi:**

| Label Sebenarnya \ Prediksi | Prediksi Non-HS (`0`) | Prediksi Hate Speech (`1`) | Total Sampel |
| :--- | :---: | :---: | :---: |
| **Aktual Non-HS (`0`)** | **976** (*True Negative*) | **151** (*False Positive*) | 1.127 baris |
| **Aktual Hate Speech (`1`)** | **151** (*False Negative*) | **673** (*True Positive*) | 824 baris |
| **Total Prediksi** | 1.127 | 824 | **1.951 baris** |

*Insight Confusion Matrix:*
- Tingkat *False Positive* sebesar **151 baris (13,40% dari total Non-HS)**: Model baseline masih memiliki sedikit kecenderungan salah mencap opini kritis atau cuitan bernada tegas sebagai ujaran kebencian.
- Tingkat *False Negative* sebesar **151 baris (18,33% dari total HS)**: Ujaran kebencian implisit atau berbasis sindiran halus belum tertangkap penuh oleh representasi leksikal TF-IDF.
- Jumlah FP dan FN yang seimbang (151 : 151) menunjukkan bahwa model tidak condong ke salah satu jenis kesalahan; ini adalah perilaku yang wajar untuk *threshold* default 0,5 dengan pembobotan kelas seimbang.

---

### 6.1.5 Analisis Koefisien Fitur & Interpretasi Linguistik

Keunggulan utama model linier adalah interpretasi langsung nilai bobot fitur ($w_j$):

| Peringkat | Top 10 Fitur Pemicu Hate Speech ($w_j > 0$) | Koefisien | Top 10 Fitur Penahan Non-HS ($w_j < 0$) | Koefisien |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `cebong` | **+5,8469** | `budaya` | **-3,5503** |
| 2 | `prabowo` | **+4,5293** | `katolik` | **-3,2415** |
| 3 | `ahok` | **+4,2307** | `bom` | **-3,0946** |
| 4 | `lengserkan` | **+4,2184** | `ekonomi` | **-3,0341** |
| 5 | `bubarkan` | **+4,1628** | `presiden jokowi` | **-2,7577** |
| 6 | `jokowi` | **+4,1152** | `m****` | **-2,3475** |
| 7 | `tolol` | **+4,1138** | `sipit` | **-2,2509** |
| 8 | `dasar` | **+4,0766** | `pilihan` | **-2,1968** |
| 9 | `ganti presiden` | **+3,8236** | `buaya` | **-2,0854** |
| 10 | `usir` | **+3,8061** | `wkwk` | **-1,9872** |

*Catatan: token vulgar pada tabel disamarkan; nilai koefisien tetap dilaporkan apa adanya.*

*Temuan Linguistik Krusial:*
1. **Dominasi Sentimen Polarisasi Politik:** Kata ejekan politik (`cebong`) menjadi prediktor kebencian terkuat (+5,85), diikuti nama figur politik (`prabowo`, `ahok`, `jokowi`) yang sering muncul pada cuitan serangan. Kata imperatif agresif (`lengserkan`, `bubarkan`, `ganti presiden`, `usir`) serta kata makian intelek (`tolol`, `dasar`, `dungu`) juga memiliki bobot positif yang besar.
2. **Konteks Unigram vs Bigram:** Unigram nama figur politik seperti `jokowi` (+4,12) memiliki asosiasi positif ke ujaran kebencian karena sering diserang pada cuitan sentimen negatif; sebaliknya bigram resmi `presiden jokowi` memiliki bobot negatif (**-2,76**), membuktikan pentingnya fitur bigram untuk membedakan pemberitaan/diskusi formal dari hujatan.
3. **Ambiguitas Leksikal & Asosiasi yang Perlu Diaudit:** Beberapa istilah identitas dan kata kasar eksplisit justru memperoleh koefisien negatif (mis. `katolik` -3,24; `sipit` -2,25; `m****` -2,35; `wkwk` -1,99). Ini tidak berarti istilah tersebut "bebas kebencian"; koefisien negatif muncul karena konteks dominan tempat istilah itu muncul di data latih (diskusi keagamaan, kutipan/reportase, atau lelucon). Temuan ini menunjukkan keterbatasan *bag-of-words* yang tidak memahami konteks, negasi, dan *reported speech*, sekaligus menandai perlunya audit fairness dan model kontekstual seperti IndoBERT.

---

### 6.1.6 Eksplorasi Sensitivitas Regularisasi ($C$) pada Validation Set

Pengujian nilai parameter regularisasi $C$ dilakukan untuk memetakan sensitivitas performa dan memastikan bahwa $C = 1.0$ adalah pilihan garis dasar yang seimbang, bukan sekadar mengikuti nilai bawaan:

| Nilai $C$ | Macro F1-Score | Precision Kelas 1 (HS) | Recall Kelas 1 (HS) | F1-Score HS | ROC-AUC | Catatan Perilaku |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `0.01` | 0,8016 | 0,8107 | 0,7172 | 0,7611 | 0,8805 | Regularisasi terlalu ketat (*Underfitting*) |
| `0.1` | 0,8279 | 0,8203 | 0,7755 | 0,7973 | 0,8987 | Recall meningkat pesat |
| `0.5` | 0,8370 | 0,8150 | 0,8070 | 0,8110 | 0,9132 | Menuju konvergensi optimal |
| **`1.0`** | **0,8414** | **0,8167** | **0,8167** | **0,8167** | **0,9168** | **Konfigurasi Garis Dasar Terpilih (Presisi Tertinggi; F1 Kompetitif)** |
| `2.0` | **0,8437** | 0,8153 | 0,8252 | 0,8203 | **0,9181** | **Macro F1 Tertinggi; Presisi Sedikit Turun** |
| `5.0` | 0,8359 | 0,8060 | 0,8167 | 0,8113 | 0,9151 | Penalti $L_2$ melemah |
| `10.0` | 0,8268 | 0,7910 | 0,8131 | 0,8019 | 0,9102 | Gejala awal *overfitting* pada kata langka |

*Kesimpulan Bab 6.1:* Model Baseline menetapkan standar performa awal yang kokoh (**Macro F1 = 84,14%**, **ROC-AUC = 91,68%**), membuktikan bahwa fitur leksikal n-gram linear sudah mampu menangkap sinyal kebencian eksplisit dengan baik. Pada eksplorasi $C$, nilai $C = 2.0$ menghasilkan Macro F1 sedikit lebih tinggi (84,37%) dan ROC-AUC 91,81%, tetapi Precision kelas HS sedikit lebih rendah (81,53% vs 81,67%). Karena Precision kelas HS adalah metrik etis yang krusial untuk mencegah penyensoran kritik, $C = 1.0$ dipertahankan sebagai konfigurasi garis dasar; $C = 2.0$ dapat dipertimbangkan pada tuning Minggu 4.

---

## BAB 6.2 — MODEL 2: ADVANCED (XGBOOST + TRUNCATED SVD + FITUR GAYA PENULISAN)
*Penanggung Jawab (PIC): Muhammad Adam Al-Kidri (244311050)*

### 6.2.1 Teori Dasar, Keunggulan & Kelemahan XGBoost

#### A. Prinsip Gradient Tree Boosting

XGBoost (*Extreme Gradient Boosting*) adalah ansambel aditif dari $K$ pohon regresi CART (*Classification and Regression Tree*). Berbeda dari *Random Forest* yang membangun pohon secara paralel dan independen, XGBoost membangun pohon secara **sekuensial**: setiap pohon baru $f_t$ dilatih untuk mengoreksi sisa kesalahan (*residual*) ansambel pohon sebelumnya. Prediksi akhir adalah jumlah skor seluruh pohon:

$$\hat{y}_i = \sum_{k=1}^{K} f_k(\mathbf{x}_i), \qquad f_k \in \mathcal{F}$$

Fungsi objektif yang diminimalkan terdiri dari kerugian pelatihan dan penalti kompleksitas pohon:

$$\mathcal{L}(\phi) = \sum_{i=1}^{N} l\left(y_i, \hat{y}_i\right) + \sum_{k=1}^{K} \Omega(f_k), \qquad \Omega(f) = \gamma T + \frac{1}{2}\lambda \|w\|^2$$

dengan $T$ = jumlah daun, $w$ = vektor bobot daun, $\gamma$ = penalti jumlah daun, dan $\lambda$ = penalti besaran bobot daun (regulerisasi $L_2$). Pada iterasi ke-$t$, XGBoost menambahkan pohon $f_t$ yang meminimalkan aproksimasi Taylor orde kedua:

$$\tilde{\mathcal{L}}^{(t)} = \sum_{i=1}^{N} \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t)$$

dengan $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ dan $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$. Bobot daun optimal dan *gain* kandidat split adalah:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \text{Gain} = \frac{1}{2}\left[ \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda} - \frac{(G_L+G_R)^2}{H_L+H_R+\lambda} \right] - \gamma$$

dengan $G_L, G_R$ = jumlah gradien anak kiri/kanan dan $H_L, H_R$ = jumlah Hessian anak kiri/kanan. Split hanya dipilih jika *gain*-nya positif. Mekanisme inilah yang membuat XGBoost mampu menangkap interaksi non-linear antar-fitur sekaligus melakukan seleksi fitur secara otomatis.

#### B. Mengapa TruncatedSVD Diperlukan Sebelum Pohon Keputusan?

Matriks TF-IDF berukuran $9.105 \times 5.000$ dengan *sparsity* **99,81%**. Pohon keputusan menguji satu fitur pada satu waktu; pada ruang 5.000 fitur *sparse*, mayoritas fitur sangat jarang dan hanya aktif pada sedikit dokumen. Akibatnya, pencarian split menjadi mahal, sinyal leksikal terpecah ke banyak kolom, dan fitur frekuensi rendah mudah menghasilkan split yang tidak general.

**TruncatedSVD** (setara *Latent Semantic Analysis* pada TF-IDF) mendekomposisi matriks $X$ menjadi:

$$X \approx U_k \Sigma_k V_k^T, \qquad Z = U_k \Sigma_k \in \mathbb{R}^{100}$$

Komponen $Z$ adalah fitur laten padat yang merangkum pola ko-okurensi kata. Reduksi ke 100 dimensi mempercepat pencarian split, meredam *noise* leksikal, dan menghasilkan representasi kontinu yang lebih ramah untuk pohon. Namun, SVD adalah transformasi linear yang **memutar** ruang fitur: 100 komponen pada data ini hanya menjelaskan **23,45%** varians TF-IDF. Artinya, sebagian besar detail leksikal halus dibuang, dan komponen SVD tidak lagi bisa dibaca langsung sebagai "kata pemicu" seperti koefisien Logistic Regression.

#### C. Feature Fusion: 100 Komponen SVD + 6 Fitur Gaya Penulisan

TF-IDF menangkap **apa yang ditulis**, tetapi tidak menangkap **bagaimana cara menulisnya**. Enam fitur gaya penulisan dari Bab 5 memberi sinyal meta-linguistik yang tidak tersedia dalam *bag-of-words*:

| Fitur | Makna | Sinyal yang Diharapkan |
| :--- | :--- | :--- |
| `char_length` | Panjang karakter | Cuitan panjang vs pendek |
| `word_count` | Jumlah kata | Kepadatan argumen |
| `caps_ratio` | Rasio huruf kapital | Emosi/teriakan |
| `exclamation_count` | Jumlah `!` dan `?` | Intensitas emosional |
| `user_mention_count` | Jumlah mention `USER` | Pola percakapan/debat |
| `abusive_lexicon_count` | Jumlah kata kasar | Sinyal leksikal eksplisit |

Penggabungan keduanya menghasilkan matriks $9.105 \times 106$ yang padat. `StandardScaler` di-*fit* hanya pada Training Set agar skala tiap fitur seragam. Meskipun pohon secara teori invarian terhadap skala monoton, standardisasi membuat perbandingan *gain* antarfitur lebih stabil dan menjaga konsistensi pipeline dengan Bab 5.

#### D. Keunggulan dan Kelemahan Dibanding Model Linear

| Aspek | Logistic Regression (Baseline) | XGBoost (Advanced) |
| :--- | :--- | :--- |
| **Hubungan fitur** | Linear pada log-odds | Non-linear, menangkap interaksi |
| **Ruang input** | 5.000 fitur *sparse* | 106 fitur padat hasil fusi |
| **Interpretasi** | Koefisien bertanda, arah jelas | *Gain importance*, tanpa arah pengaruh |
| **Optimasi** | Konveks, minimum global | Non-konveks, optimasi *greedy* |
| **Biaya latih** | Rendah | Lebih tinggi |
| **Risiko overfitting** | Terkendali oleh $L_2$ | Lebih tinggi, sensitif hiperparameter |
| **Detail leksikal** | Utuh | Terkompresi oleh SVD |
| **Fitur campuran** | Tidak natural | Natural (SVD + numerik) |
| **Kalibrasi probabilitas** | Umumnya baik | Kurang stabil pada data kecil |

Keunggulan utama XGBoost adalah kemampuannya menangkap interaksi seperti "cuitan pendek + huruf kapital tinggi + kata kasar" yang tidak dapat direpresentasikan oleh model aditif linear. Kelemahan utamanya adalah hilangnya detail leksikal akibat SVD, interpretasi yang lebih kabur, dan risiko *overfitting* pada dataset berukuran sedang (~9 ribu baris).

---

### 6.2.2 Hypothesis Space & Inductive Bias

#### A. Hypothesis Space ($\mathcal{H}$)

Ruang hipotesis XGBoost adalah keluarga fungsi aditif yang tersusun dari $K$ pohon regresi:

$$\mathcal{H} = \left\{ f(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x}) \;\middle|\; f_k \in \mathcal{F} \right\}$$

dengan setiap pohon $f_k$ didefinisikan oleh struktur partisi $q$ dan bobot daun $w$:

$$\mathcal{F} = \left\{ f(\mathbf{x}) = w_{q(\mathbf{x})} \;\middle|\; q: \mathbb{R}^{106} \to \{1,\dots,T\}, \; \mathbf{w} \in \mathbb{R}^{T} \right\}$$

Pada konfigurasi awal, $K = 100$ dan kedalaman maksimum $6$, sehingga setiap pohon memiliki paling banyak $T \le 2^6 = 64$ daun. Fungsi yang dihasilkan bersifat **piecewise constant** dan **aditif**, bukan satu hyperplane linear. Batas keputusannya merupakan gabungan kotak-kotak sejajar sumbu (*axis-aligned boxes*), bukan bidang datar tunggal seperti Logistic Regression.

#### B. Inductive Bias (Asumsi Dasar Model)

1. **Axis-Aligned Partitioning (Partisi Sejajar Sumbu):**  
   Setiap split berbentuk $x_j \le \theta$ pada satu fitur. Model mengasumsikan bahwa batas keputusan yang baik dapat didekati dengan potongan-potongan sejajar sumbu fitur. Konsekuensinya, XGBoost sangat efisien jika sinyal kelas bersifat monotonik terhadap fitur, tetapi kurang efisien jika batas keputusan miring (*oblique*) terhadap sumbu. Karena SVD memutar ruang TF-IDF, split sejajar sumbu pada 100 komponen SVD tidak selalu sejajar dengan sumbu kata asli — ini adalah salah satu alasan potensial mengapa model advanced bisa lebih lemah dari baseline linear pada teks.

2. **Hierarchical Feature Interaction (Interaksi Fitur Hierarkis):**  
   Setiap jalur dari akar ke daun dengan kedalaman $d$ dapat menguji hingga $d$ fitur secara berurutan. Dengan `max_depth=6`, satu pohon dapat merepresentasikan interaksi hingga orde 6 (mis. "kata kasar" **dan** "huruf kapital tinggi" **dan** "cuitan pendek"). Penjumlahan 100 pohon memperluas ruang interaksi ini. Inilah keunggulan utama XGBoost dibanding Logistic Regression yang hanya aditif.

3. **Regularization via Shrinkage & Tree Depth Constraint (Regularisasi melalui Penyusutan dan Batas Kedalaman):**  
   Setiap pohon baru diskalakan dengan `learning_rate=0.1` (*shrinkage*) agar kontribusinya kecil dan bertahap. Kedalaman dibatasi `max_depth=6`, jumlah pohon dibatasi 100, dan penalti $\gamma T + \frac{1}{2}\lambda\|w\|^2$ menghukum pohon yang terlalu besar. Kombinasi ini merupakan **prior parsimoni**: model diarahkan ke fungsi yang sederhana dan stabil, bukan menghafal data latih. Pembobotan kelas `scale_pos_weight=1,366` juga merupakan bentuk prior terhadap distribusi kelas yang tidak seimbang.

---

### 6.2.3 Konfigurasi Hiperparameter Awal & Hasil Validasi

#### A. Konfigurasi Hiperparameter Awal

| Komponen / Hiperparameter | Nilai Konfigurasi | Alasan Metodologis |
| :--- | :--- | :--- |
| **Varian Teks Input** | `text_clean_lexical` | Sama dengan baseline agar komparasi adil |
| **Representasi TF-IDF** | 5.000 fitur, unigram–bigram | `ngram_range=(1,2)`, `min_df=2`, `sublinear_tf=True` |
| **Reduksi Dimensi** | `TruncatedSVD(n_components=100)` | Fit hanya di train; mempercepat split dan meredam noise |
| **Fitur Gaya Penulisan** | 6 fitur numerik | Diverifikasi ulang dari `Tweet` mentah (match 100%) |
| **Standardisasi** | `StandardScaler` | Fit hanya di train; menyeragamkan skala 106 fitur |
| **Jumlah Fitur Fusi** | 106 fitur padat | 100 komponen SVD + 6 fitur gaya |
| **Algoritma** | `XGBClassifier` | Gradient boosted trees untuk interaksi non-linear |
| **Jumlah Pohon** | `n_estimators=100` | Konfigurasi awal; tuning pada Minggu 4 |
| **Kedalaman Maksimum** | `max_depth=6` | Menangkap interaksi hingga orde 6 tanpa terlalu dalam |
| **Learning Rate** | `learning_rate=0.1` | *Shrinkage* standar untuk stabilitas |
| **Pembobotan Kelas** | `scale_pos_weight=1,3662` | Rasio kelas 0 : 1 = 5.257 : 3.848 |
| **Objective** | `binary:logistic` | Klasifikasi biner probabilistik |
| **Eval Metric** | `logloss` | Kerugian logistik untuk monitoring |
| **Tree Method** | `hist` | Histogram split, efisien untuk 9.105 baris |
| **Early Stopping** | Tidak digunakan | Validation tetap murni sebagai evaluator |
| **Ambang Keputusan** | $0.5$ | Ambang probabilitas default |
| **Random Seed** | $42$ | Menjamin reprodusibilitas |

*Catatan Anti-Leakage:* `TfidfVectorizer`, `TruncatedSVD`, dan `StandardScaler` di-*fit* murni pada **9.105 baris Train Set**, lalu mentransformasi Validation Set secara pasif. Enam fitur gaya penulisan dihitung ulang dari kolom `Tweet` mentah dan dicocokkan 100% dengan kolom tersimpan.

#### B. Hasil Evaluasi pada Validation Set ($N = 1.951$)

| Metrik Evaluasi | Nilai Validasi | Target Kriteria Proyek | Status Ketercapaian |
| :--- | :---: | :---: | :---: |
| **Macro F1-Score** | **0,8222** (82,22%) | $\ge 85,00\%$ | *Belum mencapai target* |
| **Precision Kelas 1 (Hate Speech)** | **0,8055** (80,55%) | $\ge 85,00\%$ | *Belum mencapai target* |
| **Recall Kelas 1 (Hate Speech)** | **0,7791** (77,91%) | $\ge 80,00\%$ | *Belum mencapai target* |
| **F1-Score Kelas 1 (Hate Speech)** | **0,7921** (79,21%) | — | *Keseimbangan cukup baik* |
| **ROC-AUC** | **0,9034** (90,34%) | $\ge 88,00\%$ | **Melampaui Target** |
| **PR-AUC (Average Precision)** | **0,8806** (88,06%) | — | *Pemisahan kurva presisi sangat baik* |

**Matriks Kebingungan (*Confusion Matrix*) pada Data Validasi:**

| Label Sebenarnya \ Prediksi | Prediksi Non-HS (`0`) | Prediksi Hate Speech (`1`) | Total Sampel |
| :--- | :---: | :---: | :---: |
| **Aktual Non-HS (`0`)** | **972** (*True Negative*) | **155** (*False Positive*) | 1.127 baris |
| **Aktual Hate Speech (`1`)** | **182** (*False Negative*) | **642** (*True Positive*) | 824 baris |
| **Total Prediksi** | 1.154 | 797 | **1.951 baris** |

*Insight Confusion Matrix:*
- Tingkat *False Positive* sebesar **155 baris (13,75% dari total Non-HS)**: sedikit lebih tinggi dari baseline (151 baris), sehingga model advanced tidak lebih baik dalam melindungi kritik wajar.
- Tingkat *False Negative* sebesar **182 baris (22,09% dari total HS)**: lebih tinggi dari baseline (151 baris). Ini adalah penyebab utama turunnya Recall dan Macro F1: model advanced lebih banyak melewatkan ujaran kebencian implisit.
- Dengan demikian, model advanced belum memberikan trade-off yang lebih baik; ia justru lebih agresif melewatkan kelas positif tanpa mengurangi False Positive secara berarti.

#### C. Analisis Feature Importance

XGBoost menyediakan *gain importance*: total pengurangan kerugian yang disumbangkan suatu fitur ketika digunakan untuk split. Nilai ini **tidak bertanda**; ia hanya menunjukkan besarnya kontribusi, bukan arah pengaruh.

| Peringkat | Fitur | Kelompok | Gain |
| :---: | :--- | :--- | :---: |
| 1 | `SVD_008` | Komponen SVD | 0,0437 |
| 2 | `SVD_018` | Komponen SVD | 0,0301 |
| 3 | `SVD_005` | Komponen SVD | 0,0250 |
| 4 | `SVD_001` | Komponen SVD | 0,0215 |
| 5 | `user_mention_count` | **Gaya Penulisan** | 0,0214 |
| 6 | `SVD_056` | Komponen SVD | 0,0204 |
| 7 | `abusive_lexicon_count` | **Gaya Penulisan** | 0,0203 |
| 8 | `char_length` | **Gaya Penulisan** | 0,0197 |
| 9 | `SVD_017` | Komponen SVD | 0,0177 |
| 10 | `SVD_019` | Komponen SVD | 0,0171 |
| 11 | `SVD_032` | Komponen SVD | 0,0168 |
| 12 | `SVD_023` | Komponen SVD | 0,0164 |
| 13 | `SVD_036` | Komponen SVD | 0,0150 |
| 14 | `SVD_003` | Komponen SVD | 0,0148 |
| 15 | `SVD_039` | Komponen SVD | 0,0131 |

**Porsi Gain per Kelompok Fitur:**

| Kelompok | Jumlah Fitur | Total Gain | Porsi |
| :--- | :---: | :---: | :---: |
| Komponen SVD | 100 | 0,9117 | 91,17% |
| Fitur Gaya Penulisan | 6 | 0,0883 | **8,83%** |
| **Total** | **106** | **1,0000** | **100%** |

*Temuan Feature Importance:*
1. **Tiga dari enam fitur gaya penulisan masuk 15 teratas:** `user_mention_count` (peringkat 5), `abusive_lexicon_count` (peringkat 7), dan `char_length` (peringkat 8). Ini menunjukkan bahwa sinyal meta-linguistik memang kompetitif dengan komponen SVD individual.
2. **Komponen SVD tetap mendominasi (91,17% gain):** Sinyal leksikal masih menjadi pendorong utama. Fitur gaya penulisan memberi kontribusi tambahan yang nyata, tetapi belum cukup untuk mengimbangi hilangnya detail leksikal akibat kompresi SVD.
3. **Komponen SVD tidak dapat dibaca langsung:** `SVD_008`, `SVD_018`, dan `SVD_005` adalah kombinasi linear dari banyak kata. Kita tidak bisa menyebut satu kata pemicu seperti pada baseline; ini adalah harga interpretabilitas dari reduksi dimensi.
4. **Tiga fitur gaya lainnya** (`word_count`, `caps_ratio`, `exclamation_count`) berada di luar 15 teratas, tetapi tetap memberi kontribusi kecil. `caps_ratio` dan `exclamation_count` sangat *skewed* (mayoritas bernilai 0), sehingga split pohon pada fitur ini kurang informatif.

#### D. Error Analysis: False Positive & False Negative

Lima contoh False Positive dan lima contoh False Negative ditampilkan di notebook (`03_model_design_advanced_ml.ipynb`, bagian *Error Analysis*). Pola yang muncul:

- **False Positive (kritik wajar dicap hate speech):** pertanyaan hukum/politik singkat yang mengandung kata pemicu (mis. `dasar`), kutipan kata kasar di dalam narasi, serta cuitan dukungan politik dengan hashtag. Model bereaksi terhadap token permukaan dan tidak memahami negasi, kutipan, atau maksud kritik.
- **False Negative (hate speech lolos):** slur rasial dalam percakapan, berita tentang perundungan, hinaan berbasis gender, dan sindiran yang bergantung konteks. Kompresi SVD membuang detail leksikal yang sebenarnya penting, sedangkan pohon tidak memiliki mekanisme khusus untuk memahami *reported speech* atau sarkasme.
- **Perbandingan dengan baseline:** jumlah FN advanced (182) lebih tinggi dari baseline (151), sedangkan FP hampir sama (155 vs 151). Ini mengonfirmasi bahwa penurunan performa terutama terjadi pada Recall kelas HS, bukan pada Precision.

#### E. Kesimpulan Bab 6.2

Model advanced mencapai **Macro F1 = 82,22%**, **ROC-AUC = 90,34%**, dan **PR-AUC = 88,06%**. Pada konfigurasi awal Minggu 3, model ini **belum mengungguli baseline** pada Macro F1, Precision, Recall, dan ROC-AUC; satu-satunya metrik yang sedikit lebih baik adalah PR-AUC (+0,0012).

Penyebab utama yang teridentifikasi:
1. **Kompresi SVD terlalu agresif:** 100 komponen hanya menjelaskan 23,45% varians TF-IDF. Detail leksikal yang membedakan hate speech implisit dari kritik wajar banyak yang hilang.
2. **Split sejajar sumbu pada ruang SVD yang terotasi:** pohon keputusan tidak invarian terhadap rotasi fitur, sehingga split pada komponen SVD kurang efisien dibanding split pada kata asli.
3. **Konfigurasi awal belum dituning:** 100 pohon tanpa *early stopping*, `max_depth=6`, dan `learning_rate=0.1` adalah titik awal, bukan hasil optimasi.
4. **Ukuran data terbatas:** ~9 ribu baris cukup untuk model linear berdimensi tinggi, tetapi belum tentu cukup untuk ansambel pohon yang lebih kompleks.

Feature fusion tetap memberi nilai: 6 fitur gaya penulisan menyumbang **8,83% gain** dan 3 di antaranya masuk 15 teratas. Namun, kontribusi ini belum cukup untuk menutupi kehilangan informasi akibat SVD. Untuk Minggu 4, kandidat perbaikan adalah: menambah komponen SVD (200–300), *early stopping* pada validation, tuning `max_depth`/`learning_rate`, mencoba XGBoost langsung pada TF-IDF *sparse* dengan `tree_method='hist'`, serta menyetel ambang keputusan untuk menyeimbangkan Precision dan Recall.

---

## BAB 6.3 — KOMPARASI AWAL BASELINE VS ADVANCED (VALIDATION SET)
*(Disintesis bersama oleh Ardhana Syah Hidayat & Muhammad Adam Al-Kidri)*

| Metrik Evaluasi | Model 1: Baseline (Logistic Regression) | Model 2: Advanced (XGBoost + SVD + Fitur Gaya) | Selisih ($\Delta$) | Analisis Keunggulan |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1-Score** | **0,8414** | 0,8222 | **-0,0192** | Baseline unggul; SVD membuang detail leksikal |
| **Precision (Kelas 1 - HS)** | **0,8167** | 0,8055 | **-0,0112** | Baseline lebih baik menekan False Positive |
| **Recall (Kelas 1 - HS)** | **0,8167** | 0,7791 | **-0,0376** | Advanced melewatkan lebih banyak HS implisit (FN 182 vs 151) |
| **ROC-AUC** | **0,9168** | 0,9034 | **-0,0134** | Pemisahan probabilitas baseline lebih baik |
| **PR-AUC** | 0,8794 | **0,8806** | **+0,0012** | Satu-satunya metrik di mana advanced sedikit unggul |
| **Waktu Pelatihan** | **1,13 detik** | 5,70 detik | **+4,57 detik (≈5,1×)** | Advanced lebih mahal secara komputasi |

*Catatan: $\Delta$ = Advanced - Baseline. Untuk metrik performa, nilai positif berarti advanced lebih baik; untuk waktu pelatihan, nilai positif berarti advanced lebih lambat. Waktu mencakup seluruh pipeline pada lingkungan yang sama (Python 3.14, scikit-learn 1.9.1, XGBoost 3.4.1): baseline = TF-IDF + LogReg (1,13 detik), advanced = TF-IDF + SVD + scaler + XGBoost (5,70 detik).*

**Sintesis Komparasi:**

Pada konfigurasi awal Minggu 3, **model baseline linear tetap lebih unggul** pada empat dari lima metrik performa utama. Model advanced hanya sedikit unggul pada PR-AUC (+0,0012), yang menunjukkan bahwa ia mampu mempertahankan kualitas *ranking* probabilitas di sekitar *operating point* tertentu, tetapi tidak mampu mengalahkan baseline pada Macro F1, Precision, Recall, maupun ROC-AUC.

Temuan ini bukan berarti XGBoost tidak layak digunakan. Ada tiga penjelasan yang saling melengkapi:
1. **Baseline linear sangat kuat pada teks *sparse* berdimensi tinggi.** Dengan 5.000 fitur TF-IDF, Logistic Regression dapat memanfaatkan setiap token secara langsung; SVD justru memadatkan dan memutar ruang fitur sehingga sebagian sinyal hilang.
2. **Kompresi 100 komponen SVD hanya menahan 23,45% varians.** Ini adalah kompromi dimensi yang agresif. Detail leksikal yang membedakan hate speech implisit dari kritik wajar banyak yang tidak masuk ke 100 komponen.
3. **Konfigurasi advanced belum dituning.** 100 pohon tanpa *early stopping* adalah titik awal, bukan hasil optimasi. *Feature importance* menunjukkan fitur gaya penulisan memberi kontribusi nyata (8,83% gain; 3 masuk 15 teratas), tetapi belum cukup menutupi kehilangan informasi akibat SVD.

**Implikasi untuk Minggu 4 (Tuning & Perbaikan):**
- Menambah jumlah komponen SVD (mis. 200–300) untuk menahan lebih banyak varians leksikal.
- Menerapkan *early stopping* pada validation set atau split internal train.
- Melakukan *grid search* terbatas untuk `max_depth`, `learning_rate`, dan `subsample`.
- Mencoba XGBoost langsung pada matriks TF-IDF *sparse* dengan `tree_method='hist'` sebagai pembanding.
- Menyetel ambang keputusan untuk menyeimbangkan Precision dan Recall sesuai prioritas etis (mencegah penyensoran kritik).
- Membandingkan dengan IndoBERT (Model 3) yang diharapkan mampu menangkap konteks, negasi, dan sarkasme yang tidak tertangkap oleh model leksikal maupun SVD.
