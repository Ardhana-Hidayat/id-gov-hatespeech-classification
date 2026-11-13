# PROPOSAL PROYEK MACHINE LEARNING
**Judul:** Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial Menggunakan Machine Learning dan Deep Learning  
**Pengusul:** Ardhana Syah Hidayat (244311037) & Muhammad Adam Al-Kidri (244311050)

---

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang
Media sosial telah menjadi ruang publik utama bagi masyarakat Indonesia untuk memperoleh informasi, berdiskusi, serta menyampaikan tanggapan terhadap berbagai kebijakan dan penyelenggaraan pemerintahan. Namun, tingginya volume interaksi digital ini diiringi oleh maraknya penyebaran ujaran kebencian (*hate speech*) dan penggunaan bahasa kasar (*abusive language*) yang menyerang individu, kelompok, maupun institusi secara tidak proporsional. Tingginya polusi informasi berupa ujaran kebencian ini menenggelamkan kritik konstruktif dan aspirasi nyata dari masyarakat, memicu polarisasi sosial, serta menghambat terciptanya komunikasi dua arah yang sehat antara pemerintah dan warga negara.

Moderasi konten secara manual oleh manusia tidak mampu mengimbangi kecepatan dan skala produksi teks di media sosial yang mencapai ribuan unggahan per menit, serta rentan terhadap subjektivitas moderator. Di sisi lain, pemfilteran berbasis kata kunci sederhana (*keyword matching*) sering kali gagal memahami konteks kalimat bahasa Indonesia yang penuh dengan singkatan, bahasa gaul (*slang*), dan sarkasme, sehingga berisiko keliru memblokir kritik kebijakan yang sah. Oleh karena itu, penelitian ini mengembangkan dan membandingkan performa algoritma *Machine Learning* klasik, *Ensemble Learning*, dan *Deep Learning* berbasis pemrosesan bahasa alami (*Natural Language Processing* / NLP) untuk mengklasifikasikan ujaran kebencian pada wacana informasi pemerintahan Indonesia secara akurat dan kontekstual.

### 1.2 Rumusan Masalah (Problem Statements)
1. Bagaimana merancang tahapan prapemrosesan teks bahasa Indonesia (termasuk normalisasi kata tidak baku/*slang*) dan ekstraksi fitur agar mampu merepresentasikan karakteristik ujaran kebencian pada wacana publik dan pemerintahan?
2. Bagaimana perbandingan performa klasifikasi antara model *Baseline* (*Logistic Regression* + TF-IDF), model *Advanced* (*XGBoost* dengan fitur linguistik), dan model *Deep Learning* (*IndoBERT* / *Bi-LSTM*) dalam mendeteksi ujaran kebencian?
3. Bagaimana meminimalkan tingkat kesalahan *False Positive* pada model terbaik agar opini kritis masyarakat terhadap kebijakan pemerintah tidak salah diklasifikasikan sebagai ujaran kebencian?

### 1.3 Tujuan Terukur (Goals)
1. Membangun *pipeline* prapemrosesan teks bahasa Indonesia dan pemodelan klasifikasi biner yang bebas dari kebocoran data (*data leakage*) untuk ketiga arsitektur model yang diusulkan.
2. Mencapai nilai evaluasi **Macro F1-Score $\ge 85\%$** dan **ROC-AUC $\ge 88\%$** pada data uji (*test set*) menggunakan arsitektur terbaik.
3. Mencapai nilai **Precision pada kelas Ujaran Kebencian $\ge 85\%$** guna memastikan sistem memiliki keandalan tinggi dan meminimalkan risiko penyensoran terhadap kritik publik yang sah.

---

## BAB 2. TINJAUAN PUSTAKA & KETERKAITAN SDGs

### 2.1 Studi Literatur
Penelitian mengenai deteksi ujaran kebencian pada teks bahasa Indonesia telah berkembang dari pendekatan statistik kata menuju representasi kontekstual mendalam. Ibrohim dan Budi (2019) membangun korpus ujaran kebencian dan bahasa kasar pada media sosial Twitter di Indonesia yang mencakup ranah politik, pemerintahan, agama, dan ras, serta membuktikan bahwa kombinasi ekstraksi fitur *n-gram* dengan algoritma klasifikasi terarah mampu memberikan garis dasar deteksi yang kuat. Selanjutnya, Wilie et al. (2020) melalui pengembangan *benchmark* IndoNLU memperkenalkan model bahasa pra-latih **IndoBERT** yang melampaui metode *machine learning* tradisional secara signifikan karena kemampuannya menangkap konteks semantik dua arah (*bidirectional context*) pada struktur kalimat tidak baku bahasa Indonesia.

### 2.2 Keterkaitan dengan SDGs
Penelitian ini berkontribusi langsung terhadap **Sustainable Development Goal (SDG) 16: Perdamaian, Keadilan, dan Kelembagaan yang Tangguh (*Peace, Justice and Strong Institutions*)**, khususnya pada:
* **Target 16.10:** *Menjamin akses publik terhadap informasi dan melindungi kebebasan mendasar, sesuai dengan peraturan nasional dan kesepakatan internasional.*
* **Target 16.7:** *Memastikan pengambilan keputusan yang responsif, inklusif, partisipatif, dan representatif di setiap tingkatan.*

Peran *Machine Learning* dalam proyek ini adalah menyediakan sistem penyaringan cerdas (*automated content moderation & public listening filter*) pada kanal informasi publik. Dengan memisahkan ujaran kebencian destruktif dari kritik yang konstruktif, instansi publik dan pembuat kebijakan dapat menganalisis aspirasi masyarakat secara lebih jernih dan responsif, sekaligus menjaga ruang demokrasi digital Indonesia tetap aman, inklusif, dan bebas dari intimidasi siber.

---

## BAB 3. METODOLOGI & LEARNING PROBLEM

### 3.1 Deskripsi Dataset & Pertimbangan Etika
Penelitian ini menggunakan **Indonesian Abusive and Hate Speech Twitter Text Dataset** yang dikembangkan oleh Ibrohim & Budi (2019) dan tersedia secara publik di Kaggle/GitHub.
* **Ukuran & Karakteristik Data:** Terdiri dari **13.169 baris teks** berbahasa Indonesia (proporsi kelas *Hate Speech* `HS = 1` sekitar 42,2% dan *Non-Hate Speech* `HS = 0` sekitar 57,8%), dilengkapi dengan kamus normalisasi kata tidak baku (`new_kamusalay.csv` berisi >15.000 pasangan kata *slang*-baku).
* **Pembagian Data & Pencegahan *Data Leakage*:** Dataset dibagi secara *stratified* menjadi 70% *Training Set*, 15% *Validation Set*, dan 15% *Testing Set*. Proses pelatihan kosakata (*vocabulary fitting* pada TF-IDF maupun Tokenizer) hanya dijalankan pada *Training Set*.
* **Pertimbangan Etika (Bias & Privasi):** Seluruh identitas akun dan sebutan pengguna telah dianonimkan menjadi token `USER` untuk melindungi privasi. Untuk mencegah bias pembungkaman kritik (*over-censorship*), evaluasi model tidak hanya melihat akurasi total, tetapi juga mengaudit kesalahan prediksi pada kalimat yang memuat kata kunci pemerintahan/politik.

### 3.2 Perumusan Learning Problem

| Komponen | Spesifikasi Teknis |
| :--- | :--- |
| **Target ($Y$)** | Variabel biner (`HS`):<br>• $y = 1$: **Hate Speech** (Teks mengandung ujaran kebencian, hasutan, atau serangan berbasis SARA/personal pada diskursus publik/pemerintahan).<br>• $y = 0$: **Non-Hate Speech** (Teks berupa informasi netral, diskusi umum, atau kritik kebijakan yang tidak mengandung unsur kebencian). |
| **Objective** | Meminimalkan fungsi kerugian *Binary Cross-Entropy Loss* selama pelatihan dan memaksimalkan **Macro F1-Score** pada data uji, dengan penekanan khusus pada keseimbangan **Precision** dan **Recall** agar sistem mampu menyaring ujaran kebencian tanpa mengorbankan kebebasan berpendapat (*minimizing False Positives*). |
| **Data ($X$)** | Sekuens teks berbahasa Indonesia dari unggahan media sosial yang telah melalui tahapan *text cleaning*, normalisasi kamus *slang*, dan representasi numerik (vektor jarang TF-IDF untuk model ML klasik/lanjut, serta tensor *token IDs* & *attention mask* untuk model *Deep Learning*). |

### 3.3 Rencana Tiga Model Eksperimen
1. **Model 1 — Baseline: Logistic Regression dengan Fitur TF-IDF (Unigram–Bigram)**
   * **Alasan Pemilihan:** Memberikan tolok ukur dasar (*linear benchmark*) yang cepat dan transparan. Penggunaan bobot *Inverse Document Frequency* memungkinkan kita menginspeksi kata atau frasa spesifik apa yang memiliki koefisien paling kuat terhadap kelas ujaran kebencian.
2. **Model 2 — Advanced: XGBoost Classifier dengan Fitur Gabungan (TF-IDF SVD + Linguistic Features)**
   * **Alasan Pemilihan:** Mengatasi keterbatasan model linier dengan menangkap interaksi non-linier antar fitur. Selain reduksi dimensi teks (TruncatedSVD dari TF-IDF), ditambahkan fitur rekayasa perilaku penulisan seperti rasio huruf kapital (*shouting indicator*), jumlah tanda seru, panjang teks, dan proporsi kata kasar.
3. **Model 3 — Deep Learning: Transfer Learning dengan IndoBERT (atau Bidirectional LSTM)**
   * **Alasan Pemilihan:** Kata yang sama dalam bahasa Indonesia dapat bermakna kritik wajar atau ujaran kebencian tergantung pada konteks kalimatnya. Arsitektur Transformer **IndoBERT** (`indobenchmark/indobert-base-p1`) dipilih karena mekanisme *self-attention*-nya mampu memahami hubungan kontekstual antar kata secara utuh dan menangani variasi struktur kalimat di media sosial Indonesia.

---

## DAFTAR PUSTAKA
* Ibrohim, M. O., & Budi, I. (2019). Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter. In *Proceedings of the Third Workshop on Abusive Language Online* (pp. 46–57). Association for Computational Linguistics. https://doi.org/10.18653/v1/W19-3506
* Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., ... & Purwarianti, A. (2020). IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding. In *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing* (pp. 843–857).

---

## TAUTAN PROYEK (DELIVERABLES)
* **Dataset Kaggle:** https://www.kaggle.com/datasets/ilhamfp31/indonesian-abusive-and-hate-speech-twitter-text
* **Repository GitHub:** `https://github.com/Ardhana-Hidayat/id-gov-hatespeech-classification`
