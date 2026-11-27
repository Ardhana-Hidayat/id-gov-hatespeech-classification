## MINGGU 13 — [M → O] TUNING ADVANCED & DESAIN DEEP LEARNING

**Tujuan:** Model advanced teroptimasi dan model deep learning sudah berjalan end-to-end.

**Tugas:**
1. **Tuning Model 2:** Grid Search / Random Search dengan cross-validation; catat search space, best parameters, dan skor sebelum vs sesudah tuning (Bab 6.2.4).
2. Tulis kontrol overfitting pada model advanced (Bab 6.2.5).
3. Commit notebook `04_tuning_baseline_ml`. 
4. Commit notebook `04_tuning_advanced_ml`.
5. **Desain Model 3 (DL):** pilih jenis DL sesuai data (MLP/CNN/Transfer Learning/LSTM/dll.), centang jenisnya, jelaskan alasan (Bab 6.3.1). 
6. Rancang arsitektur: layer, jumlah unit/filter, inductive bias arsitektur, jumlah parameter (Bab 6.3.2). 
7. Siapkan preprocessing khusus DL: input shape, normalisasi, augmentasi (Bab 6.3.3). 
8. Implementasi dan jalankan training pertama (sanity check: loss turun, tidak error) (Bab 6.3.5).
7. Commit notebook `04_deep_learning_design` (versi awal).

**Deliverable:**
- Model 2 hasil tuning
- Model 3 berjalan end-to-end (training awal, belum final)

**Checklist selesai:**
- [ ] Tuning memakai validation/CV, bukan test set
- [ ] Arsitektur DL memenuhi syarat minimum (MLP ≥2 hidden layers, CNN ≥2 conv layers, RNN ≥1 layer, dst.)
- [ ] Training awal berjalan tanpa error dan loss menurun
- [ ] `model.summary()` tersimpan