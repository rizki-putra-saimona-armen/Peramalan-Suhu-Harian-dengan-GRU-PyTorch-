#  Peramalan Suhu Harian dengan GRU (PyTorch)

![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Selesai-brightgreen?style=flat-square)
![Series](https://img.shields.io/badge/Series-Part%205%2F6-9cf?style=flat-square)

> Bagian ke-5 dari seri notebook Machine Learning & Deep Learning — kali ini masuk ke ranah **Recurrent Neural Network**, dengan **GRU (Gated Recurrent Unit)** sebagai model utama untuk peramalan deret waktu (*time series forecasting*).

---

##  Tentang Proyek

Notebook ini membangun model **GRU** dari nol menggunakan `torch.nn.GRU`, lalu melatihnya untuk memprediksi **suhu minimum harian** berdasarkan pola historis. Fokus utamanya bukan cuma "modelnya jalan", tapi memahami:

- Bagaimana data deret waktu diubah jadi *sliding window* yang bisa dikonsumsi RNN-based model.
- Bagaimana GRU bekerja sebagai varian RNN yang lebih ringan dari LSTM.
- Kenapa panjang konteks (`seq_len`) sangat menentukan kualitas prediksi.
- Kenapa peramalan jangka panjang (*multi-step forecasting*) itu secara alami makin tidak pasti.

##  Tujuan Pembelajaran

| # | Fokus | Deskripsi |
|---|-------|-----------|
| 1 | **Custom `Dataset`** | Membuat sliding-window dataset manual untuk time series |
| 2 | **Arsitektur GRU** | Implementasi `nn.GRU` + `Linear` head untuk regresi |
| 3 | **Training Loop** | Loop training/testing dengan `Callback` dari `jcopdl` (early stopping & checkpoint) |
| 4 | **Forecasting** | Evaluasi prediksi one-step vs multi-step ke depan |
| 5 | **Analisis Ketidakpastian** | Pengaruh `n_prior` & `n_forecast` terhadap akurasi |

---

##  Dataset

| Detail | Keterangan |
|---|---|
| **Sumber** | `daily_min_temp.csv` |
| **Fitur** | `Temp` — suhu minimum harian (°C) |
| **Rentang waktu** | 1981 – 1990 (10 tahun, harian) |
| **Total data** | 3.650 baris |
| **Split** | 80% train (2.920) / 20% test (730), **tanpa shuffle** (menjaga urutan waktu) |

Dataset berupa deret waktu univariat klasik — cocok untuk melihat langsung bagaimana RNN/GRU menangkap pola musiman dan tren jangka pendek.

##  Kenapa GRU?

GRU adalah "adik" dari LSTM yang lebih sederhana: kalau LSTM punya 3 gate (*input, forget, output*) plus cell state terpisah, GRU cuma punya **2 gate** (*update* & *reset*) dan menyatukan hidden state jadi satu. Hasilnya:

-  Parameter lebih sedikit → training lebih cepat & ringan
-  Performa sering setara LSTM untuk banyak kasus, terutama dataset kecil–menengah
-  Tetap efektif meredam *vanishing gradient* dibanding RNN vanilla

Di notebook ini dibuktikan langsung: dengan `hidden_size=129` dan `num_layers=2`, model mampu turun dari MSE ~27 di epoch pertama ke ~4.76 di titik terbaiknya.

---

##  Arsitektur Model

```python
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers,
                           dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x[:, -1, :])   # hanya timestep terakhir → [batch, output_size]
        return x, hidden
```

###  Konfigurasi

| Parameter | Nilai | Catatan |
|---|---|---|
| `seq_len` | 16 | Panjang jendela konteks (16 hari ke belakang) |
| `hidden_size` | 129 | Kapasitas memori model |
| `num_layers` | 2 | Kedalaman GRU stack |
| `dropout` | 0.2 | Regularisasi antar layer |
| `batch_size` | 32 | — |
| `optimizer` | AdamW | `lr = 0.001` |
| `loss function` | MSELoss | Regresi suhu (kontinu) |

---

##  Alur Kerja (Pipeline)

```
CSV (daily_min_temp) 
   │
   ▼
Train/Test Split (80/20, no shuffle)
   │
   ▼
Custom Dataset → Sliding Window (seq_len=16)
   │
   ▼
DataLoader (batch_size=32)
   │
   ▼
Model GRU (2 layer, hidden=129)
   │
   ▼
Training Loop + Callback (early stopping, checkpoint, live plot)
   │
   ▼
Forecasting & Visualisasi (one-step & multi-step)
```

##  Hasil Training

Model dilatih dengan *early stopping* (patience) dan berhenti otomatis di **epoch 35**, dengan performa terbaik dicapai di **epoch 30**.

| Epoch | Train Cost (MSE) | Test Cost (MSE) |
|---|---|---|
| 1  | 27.19 | 16.84 |
| 5  | 6.49  | 5.55  |
| 10 | 6.25  | 4.90  |
| 15 | 6.18  | 4.80  |
| 20 | 5.99  | 4.80  |
| 25 | 5.99  | 4.80  |
| **30** | **5.93** | ** 4.76 (terbaik)** |
| 35 | 5.83  | 4.83 (stop) |

>  **Interpretasi**: MSE terbaik ≈ **4.76**, setara **RMSE ≈ 2.18 °C**. Artinya rata-rata prediksi model meleset sekitar ±2°C dari suhu aktual — cukup solid untuk model deret waktu sederhana tanpa fitur eksternal (musim, kelembapan, dll).

##  Hasil Forecasting

Notebook ini menguji model dalam beberapa skenario peramalan:

| Skenario | `n_prior` | `n_forecast` | Insight |
|---|---|---|---|
| One-step prediction | seq_len=16 | 1 langkah | Prediksi akurat, mengikuti pola aktual dengan baik |
| Multi-step (pendek konteks) | 10 | 140 | Prediksi makin meleset di langkah-langkah akhir |
| Multi-step (konteks lebih panjang) | 30 | 120 | Sedikit lebih stabil, tapi ketidakpastian tetap meningkat seiring horizon |

**Insight kunci dari eksperimen:**
- Menambah `seq_len` dari nilai default ke 16 terbukti membantu model menangkap pola lebih baik — ini menegaskan bahwa performa model deret waktu sangat bergantung pada seberapa informatif jendela konteks yang diberikan.
- Semakin jauh horizon prediksi ke masa depan, semakin besar pula ketidakpastiannya — sebuah karakteristik alami *time series forecasting*, bukan indikasi model yang gagal.
- Menambah konteks awal (`n_prior` 10 → 30) membantu, tapi tidak menghilangkan efek akumulasi error pada prediksi jangka panjang.

---

##  Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.12 |
| Deep Learning | PyTorch (`nn.GRU`, `AdamW`, `MSELoss`) |
| Training Helper | `jcopdl` (`Callback`, `linear_block`) |
| Data | Pandas, NumPy |
| Visualisasi | Matplotlib |
| Utilitas | scikit-learn (`train_test_split`), tqdm |

##  Struktur Proyek

```
├── GRU_With_Pytorch.ipynb   # Notebook utama
├── data/
│   └── daily_min_temp.csv           # Dataset suhu harian
├── model/
│   └── LSTM/                        # Checkpoint model terbaik (nama folder legacy)
└── utils.py                         # Fungsi bantu: data4pred, pred4pred
```

>  **Catatan teknis**: folder checkpoint & beberapa komentar di notebook masih memakai penamaan `LSTM` (peninggalan dari eksperimen sebelumnya), padahal arsitektur yang benar-benar dijalankan di sini adalah **GRU**. Tidak memengaruhi hasil, tapi worth di-*rename* untuk kerapian repo. 

##  Cara Menjalankan

```bash
# 1. Clone repository
git clone <repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install torch pandas numpy matplotlib scikit-learn jcopdl tqdm

# 3. Jalankan notebook
jupyter notebook "Part_5_-GRU_With_Pytorch.ipynb"
```

---

## Bagian Lain dari Seri

| Part | Topik |
|---|---|
| Part 2 | MLPClassifier — Neural Network dasar |
| Part 4 | TF-IDF & NLP pada artikel berita Indonesia |
| **Part 5** | **GRU — Peramalan deret waktu (notebook ini)** |
| Part 6 | BiLSTM — Peramalan deret waktu lanjutan |

---

