# BAB 4 — DATA UNDERSTANDING & BAB 5 — DATA PREPARATION

**Proyek:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning
**Pengusul:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)

Seluruh angka pada dokumen ini berasal dari eksekusi notebook:
- `notebooks/01_learning_problem_eda.ipynb` (Bab 4)
- `notebooks/02_data_preparation.ipynb` (Bab 5)

Keduanya dapat dijalankan ulang dari data mentah di `data/raw/` dengan `random_state = 42`.
Data hasil olahan (termasuk *test set*) disimpan di `data/processed/` dan **tidak** di-commit ke Git
(lihat `.gitignore`), sehingga dapat diregenerasi persis.

---

## BAB 4 — DATA UNDERSTANDING

### 4.1 Informasi Dataset

| Item | Keterangan |
| :--- | :--- |
| Nama dataset | *Indonesian Abusive and Hate Speech Twitter Text* (Ibrohim & Budi, 2019) |
| Sumber | Kaggle: `ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text` |
| File utama | `data.csv` — **13.169 baris × 13 kolom**, 1,77 MB, CSV, encoding `latin1` |
| File pendukung 1 | `new_kamusalay.csv` — **15.167 pasangan** kata tidak baku → baku, 0,27 MB |
| File pendukung 2 | `abusive.csv` — **125 kata** leksikon kasar, < 0,01 MB |
| Ukuran di memori | ± 3,26 MB (*deep memory usage*) |
| Bahasa | Indonesia tidak baku (media sosial/Twitter) |
| Tugas | Klasifikasi biner terarah (*supervised binary text classification*) |

**Distribusi kelas target (`HS`)**

| Kelas | Jumlah | Persentase |
| :--- | ---: | ---: |
| `0` Non-Hate Speech | 7.608 | 57,77% |
| `1` Hate Speech | 5.561 | 42,23% |
| **Total** | **13.169** | **100%** |

Rasio kelas 1,37 : 1. Ini **imbalance ringan**, bukan kasus ekstrem, sehingga metrik utama tetap
Macro F1 dan Precision kelas 1, bukan akurasi.

### 4.2 Deskripsi Fitur

| Kolom | Tipe | Deskripsi | Peran |
| :--- | :--- | :--- | :--- |
| `Tweet` | teks | Isi cuitan; `USER`/`URL` sudah dianonimkan pembuat dataset | **Input mentah** ($X_{raw}$) |
| `HS` | biner | 1 = ujaran kebencian, 0 = bukan | **Target ($Y$)** |
| `Abusive` | biner | 1 = mengandung bahasa kasar | Konteks EDA / *error analysis* — **bukan fitur** |
| `HS_Individual` | biner | Kebencian menyasar individu | Konteks — **bukan fitur** |
| `HS_Group` | biner | Kebencian menyasar kelompok | Konteks — **bukan fitur** |
| `HS_Religion` | biner | Kebencian berbasis agama | Konteks — **bukan fitur** |
| `HS_Race` | biner | Kebencian berbasis ras/etnis | Konteks — **bukan fitur** |
| `HS_Physical` | biner | Kebencian menyasar fisik/disabilitas | Konteks — **bukan fitur** |
| `HS_Gender` | biner | Kebencian berbasis gender/orientasi seksual | Konteks — **bukan fitur** |
| `HS_Other` | biner | Kebencian lain (hinaan/sindiran) | Konteks — **bukan fitur** |
| `HS_Weak` | biner | Intensitas lemah | Konteks — **bukan fitur** |
| `HS_Moderate` | biner | Intensitas sedang | Konteks — **bukan fitur** |
| `HS_Strong` | biner | Intensitas kuat | Konteks — **bukan fitur** |

**Peringatan leakage.** `Abusive` dan seluruh `HS_*` diturunkan dari anotasi kalimat yang sama dengan `HS`.
Memasukkannya sebagai fitur `X` berarti model menerima "kunci jawaban" → skor uji tinggi tetapi tidak
berguna saat inferensi. Kolom-kolom ini hanya dipakai pada EDA dan analisis kesalahan (Minggu 6).

**Fitur turunan yang dibangun pada Bab 5**

| Fitur | Jenis | Definisi | Alasan |
| :--- | :--- | :--- | :--- |
| `text_clean_contextual` | teks | Cleaning + normalisasi kamus; stopword **dipertahankan** | Input IndoBERT; negasi harus utuh |
| `text_clean_lexical` | teks | Seperti di atas, stopword umum dibuang kecuali negasi | Input TF-IDF; kurangi dimensi & noise |
| `char_length` | numerik | Jumlah karakter teks mentah | Panjang argumen vs ejekan singkat |
| `word_count` | numerik | Jumlah kata | Alternatif panjang yang tahan spasi berlebih |
| `caps_ratio` | numerik | huruf kapital / (panjang + 1) | Indikator "berteriak"/emosi |
| `exclamation_count` | numerik | Jumlah `!` dan `?` | Intensitas emosional/retoris |
| `user_mention_count` | numerik | Jumlah token `USER` | Cuitan yang menyerang akun tertentu |
| `abusive_lexicon_count` | numerik | Jumlah token cocok `abusive.csv` | Sinyal leksikal kasar tanpa membocorkan label |
| `is_gov` | biner | Memuat kata kunci pemerintahan/politik | **Hanya audit subgrup**, bukan fitur model |

### 4.3 Analisis Kondisi Data

| Aspek | Temuan | Konsekuensi |
| :--- | :--- | :--- |
| Missing values | **0** sel kosong di semua kolom | Tidak perlu imputasi |
| Duplikat teks exact | **146 baris** | Dibuang sebelum split agar tidak bocor antar-split |
| Label bertentangan | **8 tweet (28 baris)** punya `HS` berbeda untuk teks yang sama | Dibuang; label tidak reliabel |
| Token `USER` | 7.680 baris | Dinormalisasi; jumlahnya jadi fitur `user_mention_count` |
| Token `URL` / `RT` | 792 / 942 baris | Dibuang (bukan sinyal linguistik) |
| Kode hex `\xNN` | 1.435 baris | Artefak encoding emoji `latin1`; dibuang |
| Simbol/tanda baca | 12.281 baris | Dibuang; `!`/`?` dihitung terpisah sebagai fitur |
| Teks kosong | 0 baris (sebelum cleaning) | Setelah cleaning muncul **7 dokumen kosong** → dibuang |
| Outlier panjang | 24 baris (0,18%) di atas batas IQR (292 karakter); maks 561 karakter | Tidak dihapus, hanya di-*scale*; panjang > 280 menandakan cuitan gabungan/gangguan *crawling* |
| Imbalance | 42,23% : 57,77% | Cukup `class_weight='balanced'` |

**Outlier panjang teks, bukan kesalahan input.** Median 100 karakter, Q3 152, p99 256, maksimum 561.
Karena teks panjang justru sering memuat ujaran kebencian berbentuk narasi, penghapusan outlier akan
membuang informasi. Penanganan yang benar adalah *scaling* (Bab 5.4) atau pembatasan panjang token
(`max_length=128` pada IndoBERT).

### 4.4 Exploratory Data Analysis

Lima visualisasi (melebihi minimum tiga) beserta insight. Semua gambar tersedia di notebook 01.

**1) Distribusi kelas target.** 7.608 non-HS (57,77%) vs 5.561 HS (42,23%).
*Insight:* baseline "selalu tebak kelas mayoritas" sudah mencapai akurasi ±57,8%, sehingga akurasi
menyesatkan. Gunakan Macro F1 dan Precision kelas 1.

**2) Distribusi panjang tweet per kelas.** Rata-rata kelas HS **102,2** karakter (median 94), non-HS
**123,0** karakter (median 107,5); sebaran kedua kelas lebar dan saling tumpang tindih.
*Insight:* ujaran kebencian cenderung lebih pendek (ejekan singkat), sedangkan non-HS lebih banyak
berisi argumen/penjelasan. Efeknya nyata tetapi lemah → fitur pendukung, bukan penentu.

**3) Kata paling sering per kelas (stopword dibuang).**

| Kelas | 10 kata teratas |
| :--- | :--- |
| Non-HS | indonesia, gue, presiden, asing, agama, islam, daerah, jokowi, sih, gubernur |
| HS | indonesia, jokowi, presiden, cebong, partai, komunis, sih, islam, rakyat, cina |

Hanya **11,4%** token kelas HS berasal dari `abusive.csv` (kelas non-HS: 6,2%).
*Insight:* sinyal utama bukan makian, melainkan **kosakata identitas & politik** (`cebong`, `komunis`,
`kafir`, `cina`, `rezim`) ditambah sebagian kata kasar (`anjing`, `goblok`, `dungu`).
Dampak metodologis: (a) model berisiko menghukum diskusi agama/etnis yang sah → Precision kelas 1 wajib
diaudit; (b) model leksikal hanya menangkap permukaan, sehingga model kontekstual diperlukan untuk
sarkasme/sindiran. Ini **asosiasi dalam data**, bukan hubungan sebab-akibat.

**4) Rasio huruf kapital per kelas.** Rata-rata 0,155 (HS) vs 0,113 (non-HS); median 0,104 vs 0,073.
*Insight:* arah sesuai dugaan "berteriak = emosional", tetapi sebaran sangat tumpang tindih dan
banyak nilai ekstrem → fitur pelengkap, harus digabung dengan fitur teks.

**5) Proporsi Hate Speech pada tweet bertopik pemerintahan.** 4.201 tweet (31,9%) memuat kata kunci
pemerintahan/politik, dengan proporsi HS **57,61%** versus **35,02%** di luar subgrup.
*Insight:* wacana pemerintahan memang lebih "panas" dan subgrupnya cukup besar untuk diaudit pada
Minggu 6. `is_gov` tetap **tidak** dipakai sebagai fitur agar model tidak belajar "topik = kebencian".

**Batasan Bab 4.** Dataset ini adalah Twitter umum berbahasa Indonesia; label "informasi pemerintahan"
tidak diberikan oleh pembuat dataset. Karena itu cakupan pemerintahan didekati melalui subgrup kata
kunci (31,9% data) dan akan dilaporkan sebagai analisis subgrup, bukan sebagai korpus khusus pemerintah.

---

## BAB 5 — DATA PREPARATION

> **Catatan urutan.** Pada panduan, 5.3 = transformasi dan 5.4 = splitting. Di dokumen dan notebook ini
> keduanya **ditukar** menjadi 5.3 = splitting dan 5.4 = transformasi. Alasannya: `TfidfVectorizer`
> dan `StandardScaler` *belajar statistik* dari data; bila di-fit sebelum split, statistik
> validation/test ikut terpakai saat latihan — itu **data leakage**. Isi materi tetap sama, hanya
> urutan eksekusinya yang dibalik agar sah secara metodologis.

### 5.1 Data Cleaning

| Langkah | Apa | Mengapa | Bagaimana | Efek jumlah baris |
| :--- | :--- | :--- | :--- | ---: |
| Buang label konflik | 8 tweet (28 baris) berlabel `HS` ganda | Tidak ada dasar objektif memilih salah satu label; mengajarkan noise | `groupby('Tweet')['HS'].nunique() > 1` | 13.169 → 13.141 |
| Deduplikasi | 126 duplikat teks exact (dari 146 baris duplikat; 20 sisanya sudah terbuang bersama label konflik) | Teks identik di train & test membuat skor uji palsu | `drop_duplicates(subset='Tweet')` sebelum split | 13.141 → 13.015 |
| Buang dokumen kosong | 7 cuitan yang isinya hanya emoji/simbol | Dokumen tanpa sinyal hanya menambah noise | filter `text_clean_contextual == ''` | 13.015 → **13.008** |
| Mask token anonim | `USER`, `URL`, `RT` | Bukan sinyal linguistik; `USER` dihitung sebagai fitur | regex case-insensitive | — |
| Buang kode hex | `\xNN` (1.435 baris) | Artefak encoding emoji `latin1`, bukan kata | regex `\\x[0-9a-fA-F]{2}` | — |
| Buang simbol | semua non-huruf | Tidak dipakai TF-IDF; `!`/`?` sudah jadi fitur | regex `[^A-Za-z\s]` | — |
| Lowercase | semua huruf kecil | Menyatukan `Jokowi`/`jokowi` | `.lower()` | — |
| Normalisasi slang | 15.167 entri kamus, token-per-token | Ejaan tidak baku memecah sinyal yang sama | `dict` lookup O(1) | — |
| Stopword | 809 kata Sastrawi; 803 dipakai (6 kata negasi dikecualikan) | Kurangi dimensi TF-IDF | hanya pada varian *lexical* | — |

**Dua varian teks** dibuat karena kebutuhan model berbeda:

| Varian | Isi | Dipakai |
| :--- | :--- | :--- |
| `text_clean_contextual` | Bersih + normalisasi, **stopword utuh** | IndoBERT (Minggu 5) |
| `text_clean_lexical` | Seperti di atas, stopword dibuang kecuali **tidak, bukan, jangan, belum, tak, tiada, tanpa** | TF-IDF: LogReg (Minggu 3), XGBoost (Minggu 4) |

Contoh (data nyata):

| Tahap | Teks |
| :--- | :--- |
| Mentah | `RT USER: USER siapa yang telat ngasih tau elu?edan sarap gue bergaul ...` |
| Contextual | `siapa yang telat memberi tau kamu edan sarap gue bergaul ...` |
| Lexical | `telat tau edan sarap gue bergaul ...` |

Negasi dipertahankan karena "tidak setuju" dan "setuju" adalah dua makna berbeda; menghapus "tidak"
akan membalik sentimen kalimat dan merusak klasifikasi.

### 5.2 Feature Engineering

Enam fitur numerik diekstrak dari **teks mentah**, bukan teks bersih:

| Fitur | Rata-rata | Std | Median | Maks | Catatan |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `char_length` | 114,10 | 69,13 | 100 | 561 | Ekor kanan panjang |
| `word_count` | 17,25 | 10,25 | 15 | 52 | — |
| `caps_ratio` | 0,128 | 0,148 | 0,086 | 0,882 | 0 = tanpa kapital |
| `exclamation_count` | 0,519 | 1,512 | 0 | 43 | Mayoritas 0 |
| `user_mention_count` | 1,176 | 1,912 | 1 | 24 | Banyak `USER` berulang |
| `abusive_lexicon_count` | 0,677 | 0,846 | 1 | 15 | Dari lowercase **sebelum** normalisasi kamus |

Dua keputusan penting:
1. **Ekstraksi sebelum cleaning** — `caps_ratio` dan `exclamation_count` hilang jika teks sudah
   di-lowercase/dibersihkan. Ini alasan `CONTEXT.md` mewajibkan urutan ini.
2. **`abusive_lexicon_count` dari teks pra-normalisasi** — normalisasi slang dapat mengubah kata kasar
   menjadi kata baku (mis. bentuk singkatan → kata netral), sehingga fitur akan *undercount* bila
   dihitung setelah kamus diterapkan.

`is_gov` = 4.147 baris (31,88%) setelah cleaning. Profilnya: tweet pemerintahan **lebih panjang**
(131,3 vs 106,0 karakter), tanda seru lebih banyak (0,66 vs 0,45), tetapi kata kasar lebih sedikit
(0,47 vs 0,77). Artinya ujaran kebencian di ranah pemerintahan lebih sering berupa serangan/sindiran
berbasis argumen daripada makian mentah — justru kasus yang sulit bagi model leksikal dan menjadi
alasan kuat memakai IndoBERT. `is_gov` **tidak** dimasukkan ke `X`.

### 5.3 Splitting Data (70% / 15% / 15%, stratified)

Pembagian dua tahap: 70% train vs 30% sementara, lalu 30% itu dibagi dua sama besar.
`stratify=y` menjaga proporsi kelas, `random_state=42` menjamin reproduksibilitas.

| Split | Baris | Persentase | `HS=1` | Proporsi `HS=1` |
| :--- | ---: | ---: | ---: | ---: |
| Train | 9.105 | 70,0% | 3.848 | 42,26% |
| Validation | 1.951 | 15,0% | 824 | 42,23% |
| Test | 1.952 | 15,0% | 825 | 42,26% |
| **Total** | **13.008** | **100%** | **5.497** | **42,26%** |

Verifikasi yang dijalankan notebook (berupa `assert`, bukan asumsi):
- indeks ketiga split saling lepas (*disjoint*);
- tidak ada teks `Tweet` yang muncul di lebih dari satu split;
- jumlah baris ketiga split = jumlah baris total.

**Peran split.** Validation untuk pencarian hyperparameter dan *early stopping* (Minggu 3–4).
Test **disegel** dan baru dibuka sekali pada Minggu 6. Untuk menjaga integritas, `split_manifest.json`
mencatat seed, jumlah baris, dan **SHA-256 teks test** (`def61278…9b06e`), sehingga perubahan tak
sengaja pada test set bisa dideteksi.

### 5.4 Transformasi Data

| Data | Transformasi | Konfigurasi | Model |
| :--- | :--- | :--- | :--- |
| Teks *lexical* | TF-IDF unigram–bigram | `max_features=5000`, `min_df=2`, `sublinear_tf=True`, fit di train | Logistic Regression |
| Teks *lexical* | TruncatedSVD dari TF-IDF | 100 komponen, fit di train | XGBoost |
| Numerik | StandardScaler | fit di train | XGBoost (bila perlu) |
| Teks *contextual* | Tokenizer IndoBERT | `max_length=128` | IndoBERT (dijalankan di notebook model) |

Hasil di notebook: TF-IDF train **(9.105 × 5.000)**, validation (1.951 × 5.000), test (1.952 × 5.000).

**Bukti bebas leakage (dijalankan sebagai `assert`):**
- `scaler.mean_` identik dengan rata-rata **train**, dan `scaler.scale_` identik dengan std **train**;
- hasil transform train berpusat di 0 (`mean ≈ 0`);
- perbandingan dengan scaler yang (secara salah) di-fit pada seluruh data:

| Fitur | Mean train (benar) | Mean seluruh data (salah) |
| :--- | ---: | ---: |
| `char_length` | 113,8134 | 114,0969 |
| `word_count` | 17,1979 | 17,2538 |
| `caps_ratio` | 0,1291 | 0,1284 |
| `exclamation_count` | 0,5094 | 0,5188 |
| `user_mention_count` | 1,1814 | 1,1764 |
| `abusive_lexicon_count` | 0,6758 | 0,6768 |

Selisihnya kecil pada dataset ini, tetapi prinsipnya mutlak: statistik test tidak boleh masuk ke
proses latih, sekecil apa pun, karena estimasi performa menjadi terlalu optimistis dan tidak jujur.

### 5.5 Data Balancing

**Keputusan: tidak dilakukan resampling.** Alasan:
1. Rasio train 5.257 : 3.848 = **1,37 : 1** — imbalance ringan. Kelas mayoritas tidak mendominasi
   sampai merusak pembelajaran.
2. `class_weight='balanced'` sudah menjadi spesifikasi proyek dan menangani ketimpangan pada fungsi
   kerugian tanpa mengubah data.
3. Oversampling teks menggandakan dokumen; SMOTE menciptakan vektor teks sintetis yang tidak punya
   makna linguistik. Keduanya berisiko overfitting dan evaluasi menyesatkan.

Bobot kelas dihitung **hanya dari train**: `{0: 0,866, 1: 1,183}`. Validation dan test tidak diubah
jumlah maupun komposisinya. Bila nanti terbukti model bias ke kelas mayoritas (Recall kelas 1 rendah),
alternatif yang sah adalah menambah bobot kelas 1 atau menurunkan *threshold* — tetap tanpa menyentuh
val/test.

### 5.6 Ringkasan Apa–Mengapa–Bagaimana

| Langkah | Apa | Mengapa | Bagaimana |
| :--- | :--- | :--- | :--- |
| Label konflik | Buang 28 baris | Label tidak reliabel | `nunique() > 1` per teks |
| Deduplikasi | Buang 126 baris | Cegah leakage train–test | `drop_duplicates` sebelum split |
| Dokumen kosong | Buang 7 baris | Tanpa sinyal teks | filter string kosong |
| Anonim & simbol | Buang `USER`/`URL`/`RT`, hex, tanda baca | Noise non-linguistik | regex bertahap |
| Normalisasi | 15.167 entri kamus slang | Satukan variasi ejaan | dict lookup per token |
| Stopword | 803 kata, negasi dikecualikan | Kurangi dimensi tanpa membalik makna | Sastrawi − negasi |
| Fitur gaya | 6 fitur numerik dari teks mentah | Emosi/gaya hilang saat cleaning | regex & rasio |
| `abusive_lexicon_count` | Hitung kata kasar pra-normalisasi | Normalisasi dapat menghapus kata kasar | lookup `abusive.csv` |
| Fitur topik | `is_gov` hanya untuk audit | Hindari model belajar topik | keyword matching |
| Split | 70/15/15 stratified, seed 42 | Evaluasi adil & reproducible | `train_test_split` 2 tahap |
| TF-IDF | 5.000 fitur unigram–bigram | Representasi teks | fit train, transform val/test |
| Scaling | StandardScaler | Samakan skala fitur numerik | fit train, transform val/test |
| Balancing | Tanpa resampling + `class_weight` | Imbalance ringan, hindari data semu | `compute_class_weight('balanced')` |
| Test set | Disegel sampai Minggu 6 | Estimasi akhir tetap jujur | CSV + hash SHA-256 |

**Artefak keluaran**

| File | Isi |
| :--- | :--- |
| `data/processed/train.csv` | 9.105 baris, 22 kolom (teks mentah, konteks, dua varian teks, 6 fitur, `is_gov`) |
| `data/processed/val.csv` | 1.951 baris |
| `data/processed/test.csv` | 1.952 baris — **hanya dibaca pada Minggu 6** |
| `data/processed/split_manifest.json` | Seed, jumlah baris, hash teks test |

**Checklist Minggu 2**

- [x] Scaler/encoder di-fit hanya pada data training (dibuktikan `assert` di 5.4)
- [x] Test set disimpan dan tidak disentuh sampai Minggu 6 (`test.csv` + hash)
- [x] Jumlah sampel tiap split tercatat (9.105 / 1.951 / 1.952)
- [x] Setiap langkah preprocessing punya alasan tertulis (5.1–5.6)

**Keterbatasan**
1. Dataset bukan korpus khusus pemerintah; klaim "informasi pemerintahan" didekati lewat subgrup
   kata kunci (31,9%) dan akan dilaporkan sebagai analisis subgrup.
2. Anotasi tunggal dari pembuat dataset berpotensi mengandung subjektivitas; 8 tweet terbukti
   berlabel ganda.
3. Teks bercampur bahasa daerah/asing dan singkatan yang tidak semuanya ada di kamus (15.167 entri).
4. Hasil uji baru sah setelah model dilatih pada Minggu 3–6; angka di Bab 4–5 belum merupakan
   performa model.
