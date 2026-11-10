"""
Pipeline Prapemrosesan Teks dan Pembagian Dataset
Proyek: Analisis Klasifikasi Ujaran Kebencian pada Informasi Pemerintahan Indonesia di Media Sosial
Sesuai Dokumen Spesifikasi Metodologi: CONTEXT.md
"""

import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def main():
    # 0. Penentuan Path Dinamis Proyek
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent if current_dir.name == 'notebooks' else current_dir
    data_dir = root_dir / 'data'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Root Directory: {root_dir}")
    print(f"[*] Membaca data mentah dari: {data_dir}")

    # 1. Pemuatan Dataset Mentah
    df = pd.read_csv(data_dir / 'data.csv', encoding='latin1')
    kamus_df = pd.read_csv(data_dir / 'new_kamusalay.csv', encoding='latin1', header=None, names=['slang', 'formal'])
    abusive_df = pd.read_csv(data_dir / 'abusive.csv', encoding='latin1')

    print(f"[*] Total baris data mentah: {len(df)}")

    # 2. Deduplikasi Awal (Mencegah Data Leakage persis sama di Train dan Test)
    df = df.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
    print(f"[*] Total baris unik setelah deduplikasi awal: {len(df)}")

    # Inisialisasi kamus slang dan leksikon kata kasar
    slang_dict = dict(zip(kamus_df['slang'].str.strip(), kamus_df['formal'].str.strip()))
    abusive_set = set(abusive_df['ABUSIVE'].str.lower().str.strip().tolist())

    # 3. Ekstraksi Fitur Gaya Penulisan (Pre-Cleaning Feature Engineering)
    # WAJIB dilakukan sebelum teks di-lowercase atau dibersihkan
    print("[*] Mengekstrak fitur gaya penulisan teks mentah...")
    df['char_length'] = df['Tweet'].astype(str).str.len()
    df['word_count'] = df['Tweet'].astype(str).str.split().str.len()
    df['caps_ratio'] = df['Tweet'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
    )
    df['exclamation_count'] = df['Tweet'].apply(
        lambda x: sum(1 for c in str(x) if c in ['!', '?'])
    )
    df['user_mention_count'] = df['Tweet'].apply(
        lambda x: len(re.findall(r'\bUSER\b', str(x)))
    )

    # 4. Pembersihan Teks & Normalisasi Kamus Alay
    print("[*] Membersihkan dan menormalisasi teks...")

    def clean_and_normalize(text):
        text = str(text).lower()
        # Hapus token anonimisasi dan format Twitter mentah
        text = re.sub(r'\buser\b|\burl\b|\brt\b', ' ', text)
        # Hapus byte hex representasi emoji / non-ascii mentah
        text = re.sub(r'\\x[0-9a-f]{2}', ' ', text)
        # Hapus tanda baca dan karakter khusus (hanya simpan alfanumerik & spasi)
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Normalisasi kata slang -> baku dengan kamusalay
        words = [slang_dict.get(w, w) for w in text.split()]
        return ' '.join(words)

    # Versi 1: text_clean_contextual (Untuk IndoBERT Transformer - Pertahankan stopwords & negasi)
    df['text_clean_contextual'] = df['Tweet'].apply(clean_and_normalize)

    # Hitung jumlah kata kasar setelah normalisasi
    df['abusive_lexicon_count'] = df['text_clean_contextual'].apply(
        lambda x: sum(1 for w in x.split() if w in abusive_set)
    )

    # Siapkan Stopwords untuk Versi 2 (Lexical)
    # Penting: Kata negasi TIDAK boleh dihapus agar makna tidak terbalik
    sastrawi_factory = StopWordRemoverFactory()
    default_stopwords = set(sastrawi_factory.get_stop_words())
    negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'tak', 'tiada', 'tanpa'}
    lexical_stopwords = default_stopwords - negation_words

    def remove_stopwords_preserving_negation(text):
        words = [w for w in text.split() if w not in lexical_stopwords]
        return ' '.join(words)

    # Versi 2: text_clean_lexical (Untuk Model Baseline TF-IDF & XGBoost)
    df['text_clean_lexical'] = df['text_clean_contextual'].apply(remove_stopwords_preserving_negation)

    # 5. Flag Topik Pemerintahan (Untuk Error Analysis & Audit SDG 16, BUKAN fitur pelatihan X)
    gov_keywords = [
        'presiden', 'jokowi', 'prabowo', 'ahok', 'anies', 'pemerintah', 'menteri',
        'dpr', 'negara', 'rakyat', 'rezim', 'partai', 'politik', 'kpk', 'polri',
        'polisi', 'korupsi', 'pajak', 'undang', 'perppu', 'gubernur', 'pilkada', 'nkri'
    ]
    gov_regex = r'\b(?:' + '|'.join(gov_keywords) + r')\b'
    df['is_gov'] = df['text_clean_contextual'].str.contains(gov_regex, regex=True).astype(int)

    # 6. Pemilihan Kolom yang Disimpan
    # Menjaga raw tweet & label metadata untuk analisis kualitatif pasca-pemodelan
    columns_order = [
        'Tweet',
        'text_clean_contextual',
        'text_clean_lexical',
        'char_length',
        'word_count',
        'caps_ratio',
        'exclamation_count',
        'user_mention_count',
        'abusive_lexicon_count',
        'is_gov',
        'HS',
        'Abusive',
        'HS_Individual',
        'HS_Group',
        'HS_Religion',
        'HS_Race',
        'HS_Physical',
        'HS_Gender',
        'HS_Other',
        'HS_Weak',
        'HS_Moderate',
        'HS_Strong'
    ]
    df_final = df[columns_order]

    # 7. Stratified Splitting: 70% Train, 15% Validation, 15% Test
    print("[*] Membagi data secara terstratifikasi (70:15:15, random_state=42)...")
    train_df, temp_df = train_test_split(
        df_final, test_size=0.30, stratify=df_final['HS'], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['HS'], random_state=42
    )

    # 8. Simpan ke data/processed/
    train_path = processed_dir / 'train.csv'
    val_path = processed_dir / 'val.csv'
    test_path = processed_dir / 'test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n[+] Prapemrosesan data berhasil diselesaikan!")
    print(f"    - Train Set : {len(train_df)} baris (HS=1: {train_df['HS'].sum()} / {train_df['HS'].mean():.2%}) -> {train_path}")
    print(f"    - Val Set   : {len(val_df)} baris (HS=1: {val_df['HS'].sum()} / {val_df['HS'].mean():.2%}) -> {val_path}")
    print(f"    - Test Set  : {len(test_df)} baris (HS=1: {test_df['HS'].sum()} / {test_df['HS'].mean():.2%}) -> {test_path}")

if __name__ == '__main__':
    main()
