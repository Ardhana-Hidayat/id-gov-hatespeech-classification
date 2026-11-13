import os
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def main():
    # 0. penentuan path dinamis
    root_dir = Path(__file__).resolve().parent
    if root_dir.name == 'notebooks':
        root_dir = root_dir.parent
    processed_dir = root_dir / 'data' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Memproses data dari root: {root_dir}")

    # 1. pemuatan dataset mentah & kamus pendukung
    data_dir = root_dir / 'data'
    df = pd.read_csv(data_dir / 'data.csv', encoding='latin1')
    kamus_df = pd.read_csv(data_dir / 'new_kamusalay.csv', encoding='latin1', header=None, names=['slang', 'formal'])
    abusive_df = pd.read_csv(data_dir / 'abusive.csv', encoding='latin1')

    print(f"[*] Total baris mentah: {len(df)}")

    # 2. deduplikasi awal
    df = df.drop_duplicates(subset=['Tweet']).reset_index(drop=True)
    print(f"[*] Baris unik setelah deduplikasi: {len(df)}")

    # konversi kamus dan kata ke set/dict untuk lookup O(1) yang cepat
    slang_dict = dict(zip(kamus_df['slang'].str.strip(), kamus_df['formal'].str.strip()))
    abusive_set = set(abusive_df['ABUSIVE'].str.lower().str.strip())

    # 3. ekstrak gaya penulisan 
    tweet_series = df['Tweet'].astype(str)
    df['char_length'] = tweet_series.str.len()
    df['word_count'] = tweet_series.str.split().str.len()
    df['caps_ratio'] = tweet_series.apply(lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1))
    df['exclamation_count'] = tweet_series.str.count(r'[!?]')
    df['user_mention_count'] = tweet_series.str.count(r'\bUSER\b')

    # 4. fungsi pembersihan & lowercasing teks
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\b(user|url|rt)\b|\\x[0-9a-f]{2}|[^a-z0-9\s]', ' ', text)
        return ' '.join([slang_dict.get(w, w) for w in text.split()])

    df['text_clean_contextual'] = tweet_series.apply(clean_text)

    # hitung jumlah kata kasar setelah normalisasi
    df['abusive_lexicon_count'] = df['text_clean_contextual'].apply(
        lambda x: sum(1 for w in x.split() if w in abusive_set)
    )

    # 5. pembuatan stopwords leksikal
    sastrawi_stopwords = set(StopWordRemoverFactory().get_stop_words())
    negation_words = {'tidak', 'bukan', 'jangan', 'belum', 'tak', 'tiada', 'tanpa'}
    lexical_stopwords = sastrawi_stopwords - negation_words

    df['text_clean_lexical'] = df['text_clean_contextual'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in lexical_stopwords])
    )

    # 6. flag topik pemerintahan (Error Analysis & Audit SDG 16)
    gov_keywords = [
        'presiden', 'jokowi', 'prabowo', 'ahok', 'anies', 'pemerintah', 'menteri',
        'dpr', 'negara', 'rakyat', 'rezim', 'partai', 'politik', 'kpk', 'polri',
        'polisi', 'korupsi', 'pajak', 'undang', 'perppu', 'gubernur', 'pilkada', 'nkri'
    ]
    gov_regex = r'\b(?:' + '|'.join(gov_keywords) + r')\b'
    df['is_gov'] = df['text_clean_contextual'].str.contains(gov_regex, regex=True).astype(int)

    # 7. data splitting (70% Train, 15% Val, 15% Test)
    print("[*] Membagi data secara terstratifikasi (70:15:15)...")
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df['HS'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['HS'], random_state=42)

    # 8. simpan ke data/processed/
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    val_df.to_csv(processed_dir / 'val.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)

    print("\n[+] Prapemrosesan data selesai disimpan!")
    print(f"    - Train Set : {len(train_df)} baris (HS=1: {train_df['HS'].mean():.2%})")
    print(f"    - Val Set   : {len(val_df)} baris (HS=1: {val_df['HS'].mean():.2%})")
    print(f"    - Test Set  : {len(test_df)} baris (HS=1: {test_df['HS'].mean():.2%})")

if __name__ == '__main__':
    main()