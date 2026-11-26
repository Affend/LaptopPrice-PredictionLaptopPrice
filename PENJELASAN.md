# PENJELASAN: LaptopPrice

Dokumen ini adalah bahan belajar untuk presentasi dan UAS. Isinya:

1. [Gambaran sistem](#1-gambaran-sistem)
2. [Alur data dari CSV sampai tampil di website](#2-alur-data-dari-csv-sampai-tampil-di-website)
3. [Penjelasan setiap file dan fungsi](#3-penjelasan-setiap-file-dan-fungsi)
4. [Keputusan penting yang diambil dari data](#4-keputusan-penting-yang-diambil-dari-data)
5. [Hasil validasi & evaluasi (versi 1)](#5-hasil-validasi--evaluasi-versi-1)
6. [Keterbatasan](#6-keterbatasan)
7. [Daftar pertanyaan dosen beserta jawabannya](#7-daftar-pertanyaan-dosen-beserta-jawabannya)

Semua angka di dokumen ini berasal dari training versi 1 (26 November 2026, `backend/outputs/hasil_evaluasi.json`). Jika model dilatih ulang, angka terkini ada di halaman **Validasi & Evaluasi** pada website.

---

## 1. Gambaran sistem

```
                 ┌──────────────────────────────────────────────┐
 laptop_price.csv│  backend/ml  (Python + NumPy)                │
 ───────────────▶│  preprocessing → model → evaluation → train  │
                 └───────┬───────────────────────────┬──────────┘
                         │ file .joblib              │ hasil evaluasi (JSON)
                         ▼                           ▼
                  backend/models/vN/        PostgreSQL / SQLite
                         │                  (training_runs, model_versions,
                         │                   evaluation_results, predictions)
                         ▼                           │
                 ┌───────────────────────────────────┴──┐
                 │  backend/app  (FastAPI)              │
                 │  memuat model aktif saat menyala     │
                 └───────────────┬──────────────────────┘
                                 │ JSON lewat HTTP (/api/...)
                                 ▼
                 ┌──────────────────────────────────────┐
                 │  frontend (Next.js + Recharts)       │
                 │  5 halaman, semua angka dari API     │
                 └──────────────────────────────────────┘
```

Prinsipnya **satu sumber kebenaran**:
- Angka hasil hanya dihitung sekali, oleh `ml/train.py`.
- Hasil itu disimpan di database, dan website hanya menampilkan apa yang dikirim API.
- Tidak ada angka hasil yang ditulis manual di frontend. Kalimat interpretasi di halaman Validasi pun dibentuk dari angka API.

---

## 2. Alur data dari CSV sampai tampil di website

### A. Training (`python -m ml.train`)

1. **Baca CSV.** `preprocessing.muat_dataset()` membaca `backend/data/laptop_price.csv` dan menyeragamkan nama kolom (mis. `Price (Euro)` → `Price_euros`). Jika ada kolom wajib yang hilang, fungsi ini langsung memberi error yang jelas.
2. **Bersihkan.** `bersihkan_data()` membuang 1 laptop berprosesor ARM (Samsung Chromebook Plus), karena di luar cakupan model Intel/AMD.
3. **Feature engineering.** `ekstrak_fitur()` mengubah teks menjadi angka, baris per baris:
   - `parse_resolusi()` menghasilkan lebar, tinggi, Touchscreen, dan IPS.
   - `hitung_ppi()` menghitung kerapatan piksel layar.
   - `parse_memori()` menghitung kapasitas SSD, HDD, dan Flash dalam GB.
   - `kategori_cpu()`, `merek_gpu()`, dan `sederhanakan_os()` menyederhanakan kolom kategori.
   - `tambah_fitur_turunan()` membuat PPI dan versi log2 dari RAM dan kapasitas penyimpanan.
4. **Target.** `target_log()` menghasilkan y = ln(harga).
5. **Split 80/20.** `evaluation.bagi_latih_uji()` membagi data dengan `random_state = 42`: 1019 latih dan 255 uji.
6. **Pilih α Ridge.** `evaluation.pilih_alpha()` mencoba α = 0,001 … 1000 dengan 5-Fold CV di data latih.
7. **Untuk setiap model** dari `model.SPESIFIKASI_MODEL` (7 model):
   1. `validasi_silang()` menjalankan 5-Fold CV. Preprocessing di-fit ulang di setiap fold.
   2. Model dilatih ulang di seluruh data latih, lalu dievaluasi di data uji dengan `evaluasi_prediksi()` dan `verifikasi_metrik()`.
   3. Data grafik dibuat: `data_residual()`, `data_qq_plot()`, `kurva_belajar()`, `error_per_grup()`, `error_terbesar()`, `interpretasi_koefisien()`, dan `cakupan_interval()`.
8. **Analisis tambahan.**
   - Verifikasi koefisien manual vs sklearn dengan `bandingkan_koefisien()`.
   - Analisis Gradient Descent (kurva loss, batas learning rate).
   - Analisis multikolinearitas (VIF, condition number, jalur Ridge).
9. **Pilih model aktif.** `pilih_model_aturan_1se()` memilih berdasarkan CV, bukan data uji.
10. **Simpan.**
    - `db/repository.simpan_hasil_training()` membuat versi baru di `training_runs`, menyimpan 7 file `.joblib` ke `models/vN/`, mengisi `model_versions` dan `evaluation_results`, lalu menandai satu model aktif. Semua ini berjalan dalam **satu transaksi**.
    - Gambar PNG ditulis ke `outputs/` oleh `plots.ekspor_semua()`.

### B. API menyala (`uvicorn app.main:app`)

`app/main.py`, di dalam fungsi `siklus_hidup`, dijalankan sekali saat server menyala:
1. Membaca CSV lewat fungsi preprocessing yang sama, lalu menghitung ringkasan, EDA, dan pilihan dropdown. Hasilnya disimpan sebagai cache di memori.
2. Membaca baris `is_active = true` dari `model_versions`, lalu memuat file `.joblib`-nya ke memori.

### C. Pengguna memprediksi (`POST /api/predict`)

1. **Validasi input.** Pydantic (`app/skema.py`) mengecek tipe dan batas wajar, misalnya RAM 1–128 GB dan penyimpanan tidak boleh kosong. `app/rute.py` mengecek bahwa nilai kategori ada di dataset. Input yang salah dijawab dengan status 422.
2. **Bangun fitur.** `preprocessing.fitur_dari_input()` menghitung PPI dan fitur log2 dengan rumus yang **sama persis** dengan training. Kesamaan ini dijamin oleh sebuah tes.
3. **Hitung prediksi.** `layanan_prediksi.hitung_prediksi()` menghasilkan:
   - ln(harga);
   - harga euro (tingkat 2018);
   - rentang ±1,96·RMSE_log;
   - harga setelah dikali faktor inflasi dan kurs (rupiah);
   - kontribusi setiap fitur;
   - peringatan ekstrapolasi.
4. **Simpan.** `repository.simpan_prediksi()` menyimpan prediksi ke tabel `predictions`.
5. **Tampilkan.** Frontend menampilkan hasilnya dan memperbarui tabel riwayat.

### D. Website menampilkan evaluasi

1. Halaman `/evaluasi` memanggil `GET /api/models` untuk mengisi dropdown versi.
2. Halaman itu lalu memanggil `GET /api/evaluation/{id}`. Fungsi `repository.ambil_evaluasi()` menggabungkan semua `evaluation_results` milik model tersebut, perbandingan dengan model lain di run yang sama, dan ringkasan run.
3. Recharts menggambar grafik dari data itu.

---

## 3. Penjelasan setiap file dan fungsi

### 3.1 `backend/ml/` — pipeline Machine Learning

#### `config.py`
| Isi | Penjelasan |
|---|---|
| `FOLDER_*` | Lokasi folder (data, models, outputs), dihitung relatif dari file ini supaya bisa dijalankan dari mana saja. |
| `RANDOM_STATE = 42`, `UKURAN_DATA_UJI = 0.2`, `JUMLAH_FOLD = 5`, `Z_95 = 1.96` | Konstanta eksperimen. Sengaja tidak bisa diubah lewat `.env` agar hasil selalu bisa diulang. |
| `Pengaturan` | Pengaturan yang boleh diubah lewat variabel lingkungan / `.env`: `DATABASE_URL` (default SQLite), path dataset, `KURS_EUR_IDR`, `FAKTOR_INFLASI`, `CORS_ORIGINS`. |
| `info_penyesuaian_harga()` | Membaca `data/penyesuaian_harga.json`, menghitung faktor inflasi = HICP terkini ÷ rata-rata HICP 2018, mengembalikan kurs dan sumbernya. |

#### `preprocessing.py`
| Fungsi / kelas | Penjelasan |
|---|---|
| `muat_dataset(path)` | Membaca CSV, menyeragamkan nama kolom, mengecek kolom wajib. |
| `ke_angka(nilai, satuan)` | `8`, `"8GB"`, atau `"1.37kg"` menjadi float. Kolom di file ini sudah berupa angka, tetapi fungsi ini membuat pipeline tetap jalan untuk versi dataset yang belum dibersihkan. |
| `parse_resolusi(teks)` | Regex `(\d+)x(\d+)` untuk lebar & tinggi, ditambah flag Touchscreen dan IPS dari kata kuncinya. Label seperti "Full HD" diabaikan karena ada yang tidak konsisten (mis. "IPS Panel Full HD 1366x768"). |
| `hitung_ppi(l, t, inci)` | PPI = √(l² + t²) ÷ inci (Pythagoras). |
| `kategori_cpu(perusahaan, tipe)` | Intel Core i7 / i5 / i3, Intel Lainnya (Celeron, Pentium, Atom, Core M, Xeon), AMD, Lainnya. |
| `parse_memori(teks)` | Memecah dengan "+", lalu menghitung SSD, HDD, dan Flash dalam GB. 1 TB = 1000 GB, Hybrid dihitung HDD, dua drive sejenis dijumlahkan. |
| `merek_gpu()`, `sederhanakan_os()` | Intel/Nvidia/AMD; OS menjadi Windows, macOS, Linux, Tanpa OS, Chrome OS/Android. |
| `bersihkan_data(df)` | Membuang baris di luar cakupan (CPU bukan Intel/AMD), dengan alasan tercatat. |
| `ekstrak_fitur(df)` | Menjalankan semua parser per baris. Tidak "mempelajari" apa pun, sehingga aman dari leakage. |
| `tambah_fitur_turunan(fitur)` | PPI, `Ram_log2 = log2(RAM)`, `SSD_log2 = log2(1 + SSD_GB)`, dan seterusnya. Dipakai untuk CSV **dan** form website. |
| `target_log(df)` | y = ln(harga). Alasannya ditulis di docstring (lihat juga bagian 4). |
| `fitur_dari_input(data)` | Membangun 1 baris fitur dari input API. |
| `Preprocessor` | **`fit()` hanya di data latih**: daftar kategori (yang muncul < 5 kali di data latih digabung menjadi "Lainnya"), kategori acuan (yang terbanyak), rata-rata & std numerik, rentang data latih, dan deteksi kolom one-hot identik. **`transform()`**: standardisasi z = (x − μ)/σ, fitur biner apa adanya, one-hot tanpa kolom acuan (drop-first), serta suku polinomial bila diminta. `fitur_asal()` memetakan nama kolom ke fitur asalnya untuk grafik kontribusi. |

#### `model.py`
| Kelas / fungsi | Penjelasan |
|---|---|
| `tambah_kolom_bias(X)` | Menambah kolom angka 1 (untuk intersep w₀). |
| `RegresiLinearNormalEquation` | `w = pinv(XᵀX) · Xᵀy`. Pseudo-inverse tetap bekerja walau XᵀX singular. |
| `RegresiLinearGradientDescent` | Iterasi `w ← w − η · (2/n) Xᵀ(Xw − y)`. Riwayat MSE disimpan di `riwayat_loss_`. Berhenti jika penurunan loss < toleransi. Memberi error yang jelas jika divergen. |
| `RegresiRidgeManual` | `w = (XᵀX + αI')⁻¹ Xᵀy`. I' = identitas dengan elemen intersep 0, artinya intersep tidak dipenalti (sama dengan sklearn). |
| `BaselineRataRata` | Selalu menebak rata-rata harga euro data latih. Disimpan sebagai ln(rata-rata harga) agar exp() tepat sama dengan rata-rata, bukan rata-rata geometrik. |
| `ModelHargaLaptop` | Pembungkus **Preprocessor + regresor**. Karena digabung, setiap `fit()` (termasuk di tiap fold CV) mem-fit ulang preprocessing. `predict_log()`, `predict_euro()`, dan `kontribusi()` (sumbangan wⱼ·xⱼ per fitur relatif terhadap laptop acuan). |
| `SPESIFIKASI_MODEL`, `buat_model()` | Daftar 7 model yang dibandingkan, dan pembuat model baru (belum dilatih) berdasarkan nama. |

#### `evaluation.py`
| Fungsi | Penjelasan |
|---|---|
| `mae`, `mse`, `rmse`, `mape`, `r2` | Metrik **manual** dengan NumPy. |
| `hitung_metrik_sklearn`, `verifikasi_metrik` | Metrik yang sama dihitung dengan `sklearn.metrics`, lalu dibandingkan baris per baris. |
| `bagi_latih_uji(n)` | `train_test_split` 80/20, `random_state = 42`. |
| `evaluasi_prediksi(y_log, ŷ_log)` | exp() kedua nilai, lalu menghitung metrik di skala euro, plus R² dan RMSE di skala log. |
| `validasi_silang(pembuat, F, y, k=5)` | K-Fold (shuffle, `random_state` tetap). Skor validasi & latih per fold, ditambah rata-rata dan std (ddof = 1). |
| `pilih_alpha()` | Mencoba setiap α, lalu memilih RMSE CV terkecil. |
| `kurva_belajar()` | Melatih dengan 10%…100% data latih tiap fold, lalu mencatat RMSE latih vs validasi. |
| `cakupan_interval()` | Menguji rentang ±1,96·RMSE_log: berapa persen harga aktual uji yang benar-benar di dalam rentang. |
| `data_residual()` | Titik prediksi-aktual, residual log & euro, histogram + kurva normal, Shapiro-Wilk, skewness, kurtosis, korelasi \|residual\|–prediksi. |
| `data_qq_plot()` | Kuantil residual terstandar vs kuantil normal teoretis. |
| `error_per_grup()`, `error_terbesar()` | MAE / MAPE / bias per merek atau tipe, dan 10 error terbesar. |
| `interpretasi_koefisien(model)` | (e^w − 1)·100% per fitur, ditambah efek "per penggandaan" untuk fitur log2. |
| `bandingkan_koefisien()` | Selisih maksimum bobot & prediksi antara dua regresor. |
| `hitung_vif()`, `nilai_eigen()`, `condition_number()` | Analisis multikolinearitas. |

#### `train.py`
| Isi | Penjelasan |
|---|---|
| `HYPERPARAMETER_GD` | lr = 0,1, maksimum 20.000 iterasi, toleransi 1e-14. |
| `KANDIDAT_ALPHA`, `ALPHA_JALUR_RIDGE`, `LEARNING_RATE_PEMBANDING` | Grid percobaan. |
| `analisis_gradient_descent()` | Nilai eigen Hessian, batas lr stabil 2/λmax, kurva loss 5 learning rate. |
| `analisis_multikolinearitas()` | VIF, condition number di berbagai skenario, jalur koefisien Ridge. |
| `KOMPLEKSITAS_MODEL`, `pilih_model_aturan_1se()` | Pemilihan model aktif (lihat bagian 4.6). |
| `jalankan_pipeline()` | Menjalankan semua langkah dan mengembalikan `(hasil, model_terlatih)`. |
| `simpan_model()` | Menulis `.joblib` (model + RMSE log CV untuk rentang prediksi). |
| `main()` | CLI: `--no-db` (tanpa database), `--tanpa-gambar`. |

#### `plots.py`
`ekspor_semua()` membuat 12 gambar PNG di `outputs/` untuk laporan: distribusi harga, korelasi, perbandingan model, prediksi vs aktual, residual, histogram residual, Q-Q, learning curve, kurva loss GD, error per merek, koefisien, dan pemilihan α.

### 3.2 `backend/db/` — database

| File | Penjelasan |
|---|---|
| `database.py` | `buat_engine(url)` membuat engine. Untuk SQLite, `PRAGMA foreign_keys=ON` dinyalakan agar *cascade delete* bekerja seperti di PostgreSQL. `ambil_sessionmaker()` membuat sesi database. |
| `tabel.py` | 4 tabel ORM. **training_runs** (id = nomor versi, ringkasan JSON tingkat-run), **model_versions** (id, training_run_id, nama_model, label, hyperparameter JSON, path_file, dibuat_pada, is_active), **evaluation_results** (id, model_version_id, jenis, data JSON; unik per model+jenis), **predictions** (id, model_version_id, input JSON, harga_prediksi_eur, harga_prediksi_idr, dibuat_pada). JSON = JSONB di PostgreSQL. |
| `repository.py` | Semua query. `simpan_hasil_training()` (satu transaksi, matikan semua `is_active` lalu nyalakan satu), `ambil_model_aktif()`, `daftar_model()`, `ambil_evaluasi()`, `simpan_prediksi()`, `riwayat_prediksi()`, dan `bersihkan_json()` (NaN → null, karena PostgreSQL menolak NaN). |

`evaluation_results.jenis` berisi: `metrik`, `verifikasi`, `cv`, `residual`, `qq_plot`, `learning_curve`, `loss_history` (hanya model GD), `error_per_merek`, `error_per_tipe`, `top_error`, `koefisien`.

### 3.3 `backend/alembic/` — migrasi

`alembic upgrade head` membuat atau memperbarui tabel sesuai file di `alembic/versions/`. `env.py` mengambil alamat database dari `DATABASE_URL` yang sama dengan API. Mode *batch* dipakai agar migrasi juga jalan di SQLite.

### 3.4 `backend/app/` — FastAPI

| File | Penjelasan |
|---|---|
| `main.py` | `buat_app()`: CORS, format pesan galat validasi, dan `siklus_hidup` (memuat dataset & model aktif sekali saat menyala). Jika model gagal dimuat, API tetap hidup dan `/predict` menjawab 503. |
| `rute.py` | 9 endpoint (daftar ada di README). Isinya tipis: memanggil layanan/repository saja. |
| `skema.py` | `InputPrediksi` (Pydantic: tipe & batas wajar; validator "penyimpanan tidak boleh kosong"), `ke_kolom_model()` (nama field API → nama kolom pipeline), dan bentuk respons. |
| `layanan_dataset.py` | `ringkasan()`, `eda()`, `opsi_form()`: semuanya dihitung dari CSV memakai fungsi preprocessing yang sama. Untuk form yang mudah dipakai: `daftar_model_cpu()` (92 model prosesor beserta kategori & clock yang paling umum, sehingga pengguna tidak perlu mengetik GHz), `daftar_model_gpu()` (105 model kartu grafis → merek), `berat_tipikal()` (median berat per tipe & ukuran layar), `daftar_laptop()` + `cari_laptop()` (cari laptop di dataset untuk mengisi form; ditandai data latih/uji sesuai pembagian training, data uji ditampilkan lebih dulu). |
| `layanan_prediksi.py` | `muat_model_aktif()`, `hitung_prediksi()` (harga, rentang, inflasi & kurs, kontribusi), `_peringatan()` (input di luar rentang data latih, merek langka, kolom identik). |

### 3.5 File lain di `backend/`

| File | Penjelasan |
|---|---|
| `siapkan.py` | Dijalankan otomatis di Docker: tunggu database, migrasi, lalu training jika belum ada model aktif atau file modelnya hilang. |
| `tests/` | 72 tes pytest (lihat README). `conftest.py` menjalankan pipeline sekali per sesi tes dan membuat database SQLite sementara lewat migrasi Alembic. |
| `analysis.ipynb` | Notebook analisis langkah demi langkah dengan penjelasan Markdown. |
| `data/penyesuaian_harga.json` | 12 nilai HICP bulanan 2018, HICP Oktober 2026, kurs, dan sumber masing-masing. |

### 3.6 `frontend/`

| File | Penjelasan |
|---|---|
| `lib/api.ts` | `API_URL` (dari `NEXT_PUBLIC_API_URL`), `ambilJson()` / `kirimJson()`, `GalatApi` (status 0 = backend mati), hook `useApi()` (status memuat, galat, `muatUlang`). |
| `lib/format.ts` | Format angka Indonesia: Rupiah, euro, persen, notasi ilmiah, tanggal. |
| `lib/tipe.ts` | Tipe TypeScript untuk semua respons API. |
| `lib/warna.ts` | Palet grafik yang sudah divalidasi aman untuk buta warna, ditambah warna heatmap korelasi. |
| `components/ui.tsx` | Kartu, StatTile, `Memuat`, `PesanGalat` (berisi petunjuk menyalakan backend + tombol *Coba lagi*), `KotakInfo`. |
| `components/grafik.tsx` | Pembungkus Recharts: histogram (+ kurva normal), batang horizontal (divergen), batang berkelompok, sebar (garis y=x / diagonal / nol), garis (skala log), heatmap korelasi. `ticksLog()` membuat tick sumbu log sendiri. |
| `components/Navigasi.tsx` | Menu dan indikator status backend (diperiksa ulang tiap 10 detik). |
| `components/Rumus.tsx` | Merender LaTeX dengan KaTeX di server. |
| `app/page.tsx` | Beranda: latar belakang, tujuan, diagram alur, angka ringkas dari API. |
| `app/eksplorasi/page.tsx` | Statistik dataset, distribusi harga vs ln(harga), korelasi, harga per kelompok, sebaran, contoh data. |
| `app/prediksi/page.tsx` | (1) **Cari laptop di dataset** lalu klik *Pakai* untuk mengisi form sekali klik. (2) Form: prosesor & kartu grafis dipilih berdasarkan **nama model** (kategori CPU, clock GHz, dan merek GPU terisi otomatis), resolusi diberi nama umum (HD, Full HD, 4K), berat terisi otomatis dari berat tipikal. (3) Hasil: harga Rp & kisaran, langkah perhitungan, peringatan, kontribusi fitur, dan **perbandingan dengan harga asli** bila laptop diambil dari dataset. (4) Riwayat. |
| `app/evaluasi/page.tsx` | 14 bagian validasi (lihat bagian 5), dengan dropdown versi model. |
| `app/metode/page.tsx` | Rumus dan penjelasan metode (KaTeX). |

### 3.7 Docker

| File | Penjelasan |
|---|---|
| `docker-compose.yml` | `postgres` (dengan healthcheck), `backend` (menunggu postgres sehat; volume untuk model & output), `frontend`. |
| `backend/Dockerfile` | `python:3.12-slim`, pasang requirements, jalankan sebagai user biasa, `CMD`: `siapkan.py` lalu `uvicorn`. |
| `frontend/Dockerfile` | Build multi-tahap: `npm ci` → `npm run build` (dengan `NEXT_PUBLIC_API_URL`) → image kecil berisi output *standalone*. |

---

## 4. Keputusan penting yang diambil dari data

### 4.1 Format dataset berbeda dari deskripsi tugas
Isi file **dibaca dulu** sebelum parser ditulis. Hasilnya: RAM, berat, dan GHz sudah berupa angka, CPU sudah dipecah menjadi `CPU_Company`, `CPU_Type`, dan `CPU_Frequency (GHz)`, dan target bernama `Price (Euro)`. Data berisi 1275 baris, 15 kolom, tanpa nilai kosong, dan tanpa duplikat penuh.

### 4.2 Target ln(harga)
Skewness harga **1,51**, sedangkan skewness ln(harga) **−0,18**. Selain itu efek spesifikasi bersifat persentase, dan exp() menjamin harga positif.

### 4.3 Kolinearitas sempurna terdeteksi
Pada percobaan pertama, matriks XᵀX **singular** (rank 37 dari 39 kolom). Ada dua penyebab:
- **Company=Apple ≡ OS=macOS.** Ke-21 laptop Apple semuanya macOS, dan semua macOS adalah Apple.
- **CPU=Lainnya ≡ GPU=Lainnya.** Keduanya hanya berisi 1 laptop ARM yang sama.

Solusinya: kolom one-hot identik dibuang otomatis (OS=macOS, sehingga efek Apple = merek + macOS), dan laptop ARM dikeluarkan. Hasilnya **full rank 35/35** dengan condition number **748**.

### 4.4 Lebar & tinggi resolusi tidak dipakai
| Fitur | VIF (semua fitur) | VIF (fitur dipakai) |
|---|---|---|
| Lebar_px | 125,2 | dibuang |
| Tinggi_px | 124,6 | dibuang |
| PPI | 130,2 | 1,7 |
| Inches | 17,0 | 3,8 |

Setelah lebar & tinggi dibuang (informasinya sudah ada di PPI), VIF tertinggi turun menjadi 3,8 dan condition number dari 4.119 menjadi 748.

### 4.5 RAM & penyimpanan dalam log2
Analisis error menemukan laptop RAM 64 GB ditebak €9.749, padahal harga sebenarnya €3.975. Penyebabnya, model linear mengekstrapolasi kenaikan harga per GB secara lurus.

Dengan log2, setiap **penggandaan** kapasitas menaikkan harga dengan persentase yang sama. Keputusan ini diambil **dari 5-Fold CV di data latih saja**: RMSE CV regresi linear turun dari **389,1** menjadi **359,8**.

Catatan kejujuran: ide perbaikan ini muncul dari melihat error data uji. Karena itu kriteria utamanya adalah CV, dan skor uji akhir mungkin sedikit optimis.

### 4.6 Aturan 1-SE untuk memilih model aktif
- RMSE CV terkecil dimiliki model polinomial (351,55).
- Standard error-nya = 61,71 ÷ √5 = **27,60**, sehingga batas 1-SE = **379,14**.
- Keenam model berada di bawah batas, jadi dipilih yang **paling sederhana**: regresi linear (Normal Equation manual). Ini sesuai judul tugas, dan koefisiennya mudah ditafsirkan.

Bukti bahwa keunggulan polinomial tidak meyakinkan:
- Polinomial hanya unggul di 3 dari 5 fold.
- Celah overfitting-nya lebih besar: RMSE latih 331 vs uji 447.
- Residualnya tidak normal (Shapiro p < 0,001).
- Di data uji, RMSE-nya justru lebih buruk (446,8 vs 433,4).

### 4.7 Penyesuaian harga ke kondisi sekarang
Kaggle mengunggah file ini tahun 2024, tetapi isinya laptop generasi 2016–2018. Model **dilatih & divalidasi pada harga asli** dataset, karena hanya itu yang punya harga aktual untuk dicocokkan. Hasil prediksi lalu dikali faktor inflasi zona euro **1,302** dan kurs **Rp20.010/€**.

Faktor ini tidak mengubah MAPE dan R² (semua harga dikali bilangan yang sama), dan hanya menskalakan MAE/RMSE.

### 4.8 Hal kecil lain
- `Product` dibuang karena 617 nilai unik. Model hanya akan menghafal nama produk.
- Kategori yang muncul kurang dari 5 kali di data latih digabung menjadi "Lainnya" (mis. Razer, Xiaomi, Huawei).
- "Flash Storage" dipisah dari SSD (umumnya eMMC 16–64 GB yang murah).
- Android (2 laptop) digabung dengan Chrome OS.

---

## 5. Hasil validasi & evaluasi (versi 1)

### 5.1 Perbandingan model
| Model | RMSE CV (€) | MAPE CV | R² CV | RMSE uji | MAE uji | MAPE uji | R² log uji |
|---|---|---|---|---|---|---|---|
| Baseline rata-rata | 680,0 ± 43,6 | 71,45% | −0,007 | 777,0 | 537,6 | 67,96% | −0,064 |
| **Linear: Normal Equation (aktif)** | **359,8 ± 57,1** | **20,58%** | **0,718** | **433,4** | **244,7** | **19,89%** | **0,831** |
| Linear: Gradient Descent | 359,8 ± 57,1 | 20,58% | 0,718 | 433,4 | 244,7 | 19,89% | 0,831 |
| Linear: scikit-learn | 359,8 ± 57,1 | 20,58% | 0,718 | 433,4 | 244,7 | 19,89% | 0,831 |
| Ridge manual / sklearn (α = 1) | 358,7 ± 55,6 | 20,56% | 0,720 | 432,3 | 245,2 | 19,94% | 0,832 |
| Polinomial derajat 2 | 351,5 ± 61,7 | 19,21% | 0,731 | 446,8 | 229,6 | 17,73% | 0,845 |

Dibanding baseline, model aktif menurunkan MAE 54,5%, RMSE 44,2%, dan MAPE 70,7% di data uji.

### 5.2 Verifikasi perhitungan manual
| Perbandingan | Selisih maksimum bobot |
|---|---|
| Normal Equation vs `LinearRegression` | 1,8 × 10⁻¹³ |
| Gradient Descent vs Normal Equation | 2,6 × 10⁻⁵ (konvergen setelah 9.123 iterasi) |
| Ridge manual vs `Ridge` | 1,5 × 10⁻¹⁴ |
| Polinomial manual vs `LinearRegression` | 7,0 × 10⁻¹³ |

MAE, RMSE, MAPE, dan R² manual sama persis dengan `sklearn.metrics` (selisih 0).

### 5.3 Diagnostik model aktif (data uji)
- **Residual ln(harga):**
  - rata-rata 0,007, std 0,257;
  - skewness 0,27, excess kurtosis 0,76;
  - Shapiro-Wilk W = 0,990 dengan **p = 0,090**: tidak ada bukti residual tidak normal pada α = 5%;
  - korelasi |residual| dengan prediksi 0,11: lemah, jadi variansi error cukup konstan.
- **Rentang ±1,96·RMSE_log** (RMSE_log CV = 0,261): **94,1%** harga uji berada di dalam rentang (target 95%; 15 dari 255 di luar). Rentangnya −40% … +67% dari harga prediksi.
- **Learning curve:** dari 82 ke 816 data, RMSE validasi turun dari 446 ke 360 dan mendatar di akhir, sementara RMSE latih naik ke 345. Celahnya kecil (sekitar 15 euro), jadi tidak overfitting. Menambah data sejenis tidak banyak membantu; yang dibutuhkan adalah fitur yang lebih informatif.
- **Gradient Descent:** λmax Hessian = 6,20, sehingga batas lr stabil = **0,322**. Pada lr = 0,34, loss terbukti meledak.

### 5.4 Error per tipe (data uji)
| Tipe | n | MAPE | Bias rata-rata |
|---|---|---|---|
| Notebook | 148 | 17,7% | −€56 |
| Gaming | 42 | 23,7% | −€260 (terlalu murah) |
| Ultrabook | 31 | 20,7% | +€69 |
| 2 in 1 Convertible | 20 | 23,7% | +€75 |
| Workstation | 9 | 16,0% | −€233 (terlalu murah) |
| Netbook | 5 | 40,1% | −€57 |

Kelima error terbesar semuanya **ditebak terlalu murah**: Razer Blade Pro (€6.099 → €2.934), Lenovo ThinkPad P51 (€4.899 → €1.855), HP ZBook 17, dan dua Asus ROG. Nilai yang membuat laptop ini mahal (GPU kelas atas, seri premium, merek langka) tidak terekam di fitur. Contohnya, GPU hanya dicatat mereknya, bukan tipenya.

### 5.5 Interpretasi koefisien (model aktif)
Efek dihitung dengan fitur lain tetap.

**Menaikkan harga:**
- Workstation +79,1% dibanding Notebook.
- Ultrabook +31,8%.
- 2 in 1 +22,5%.
- **RAM: setiap 2× lipat +28,0%.**
- Apple (termasuk macOS) +19,2% dibanding Dell.
- PPI +9,7% per +44,5 PPI.
- SSD: setiap 2× lipat +3,8%.

**Menurunkan harga:**
- CPU AMD −33,5% dibanding Intel Core i7.
- Intel Lainnya (Celeron/Pentium/Atom) −32,6%.
- Tanpa OS −25,1% dibanding Windows.
- Intel Core i3 −21,2%.
- Acer −15,2% dibanding Dell.
- Linux −13,8%.

---

## 6. Keterbatasan

1. **Data lama.** Harga pada tingkat sekitar 2018 (CPU Intel gen 6–8, GTX 10-series). Penyesuaian inflasi tidak sama dengan harga pasar laptop baru sekarang. Untuk itu dibutuhkan data harga terbaru.
2. **Fitur terbatas.** Tipe GPU, generasi CPU (mis. 7200U vs 8250U), dan kualitas build tidak dimodelkan, sehingga laptop premium sering ditebak terlalu murah.
3. **Jumlah data kecil.** Hanya 1.274 laptop, dan beberapa merek hanya punya 2–7 data.
4. **Rentang lebar.** −40% … +67%. Ini gambaran jujur ketidakpastian model yang hanya melihat spesifikasi.
5. **Pemilihan α Ridge.** α dipilih dengan CV di seluruh data latih, lalu CV yang sama dipakai untuk melaporkan skor Ridge (bukan *nested CV*), jadi skor Ridge sedikit optimis. Dampaknya kecil karena kurvanya datar.

---

## 7. Daftar pertanyaan dosen beserta jawabannya

### A. Konsep & metode

**1. Mengapa memakai regresi, bukan klasifikasi?**
Yang diprediksi adalah harga, yaitu nilai kontinu (angka), bukan kelas/label. Regresi memodelkan hubungan fitur → nilai kontinu.

**2. Apa itu regresi linear berganda?**
Model ŷ = w₀ + w₁x₁ + … + wₚxₚ: harga ditebak sebagai jumlah berbobot banyak fitur sekaligus. "Berganda" karena fiturnya lebih dari satu. Di sistem ini ada 34 kolom fitur setelah one-hot.

**3. Mengapa targetnya ln(harga)?**
Ada empat alasan:
- Harga miring ke kanan (skewness 1,51, menjadi −0,18 setelah log).
- Efek spesifikasi bersifat persentase, sehingga menjadi aditif di skala log.
- Variansi error lebih seragam.
- exp(ŷ) selalu positif.

Konsekuensinya, koefisien dibaca sebagai (e^w − 1)·100%.

**4. Jelaskan Normal Equation.**
MSE adalah fungsi kuadrat (bentuk mangkuk). Turunannya disamakan dengan nol: −2Xᵀ(y − Xw) = 0, sehingga XᵀXw = Xᵀy dan w = (XᵀX)⁻¹Xᵀy. Kode memakai pseudo-inverse (`np.linalg.pinv`, lewat SVD) agar tetap aman jika XᵀX singular.

**5. Kenapa masih perlu Gradient Descent kalau sudah ada Normal Equation?**
- Untuk menunjukkan bahwa keduanya mencari minimum yang sama dengan cara berbeda: solusi langsung vs iteratif.
- Untuk data sangat besar, menghitung (XᵀX)⁻¹ itu mahal (O(p³)) dan GD lebih praktis.
- GD juga dasar dari hampir semua algoritma pembelajaran modern.

Di sistem ini GD konvergen ke bobot Normal Equation dengan selisih maksimum 2,6 × 10⁻⁵.

**6. Bagaimana memilih learning rate? Apa yang terjadi jika terlalu besar?**
Untuk MSE, GD stabil jika η < 2/λmax, dengan λmax nilai eigen terbesar dari (2/n)XᵀX. Di data ini λmax = 6,20, sehingga batasnya 0,322. Kami memakai 0,1.
- η = 0,01: masih jauh dari minimum setelah 3000 iterasi.
- η = 0,3: konvergen paling cepat.
- η = 0,34: loss meledak.

Semuanya terlihat di grafik kurva loss.

**7. Kenapa fitur harus distandardisasi?**
- Supaya skala fitur seragam. Tanpa itu, RAM (2–64) dan SSD (0–1000) membuat "mangkuk" loss sangat lonjong, sehingga GD lambat atau tidak stabil.
- Koefisien antar-fitur numerik bisa dibandingkan, karena satu satuan = satu standar deviasi.
- Ridge memerlukan skala seragam agar penaltinya adil.

**8. Apa itu data leakage dan bagaimana mencegahnya?**
Data leakage terjadi jika informasi data uji ikut dipakai saat melatih, sehingga skor terlalu optimis. Pencegahannya:
- `Preprocessor.fit()` (rata-rata, std, daftar kategori) hanya dipanggil pada data latih, dan di dalam CV di-fit ulang di setiap fold.
- Data uji tidak dipakai untuk memilih α maupun model.
- Ini dijamin oleh tes `test_preprocessor_fit_hanya_dari_data_latih`.

**9. Apa itu one-hot encoding dan dummy variable trap?**
- **One-hot** mengubah kategori menjadi kolom 0/1.
- **Dummy variable trap**: jika semua kategori diberi kolom, jumlah kolomnya selalu 1 untuk setiap baris, sama persis dengan kolom intersep. Akibatnya terjadi kolinearitas sempurna dan bobot tidak unik.
- Solusinya **drop-first**: satu kategori acuan (yang terbanyak, mis. Dell) tidak diberi kolom. Bobot merek lain dibaca sebagai selisih terhadap Dell.

**10. Apa itu Ridge dan kenapa α terpilih kecil?**
- Ridge menambah penalti α·Σw² agar bobot tidak terlalu besar saat fitur saling berkorelasi.
- α terpilih = 1, dan kurva RMSE CV datar untuk α ≤ 10. Artinya setelah fitur kolinear (lebar/tinggi) dibuang dan condition number tinggal 748, regularisasi hampir tidak diperlukan.
- Peran Ridge terlihat jelas pada skenario semua fitur: jalur koefisien lebar, tinggi, dan PPI menyusut dan stabil saat α membesar.

**11. Apakah regresi polinomial masih termasuk regresi linear?**
Ya. Polinomial "linear terhadap bobot". Fiturnya saja yang ditambah (x², x·y), lalu diselesaikan dengan Normal Equation yang sama.

### B. Validasi & evaluasi

**12. Mengapa data dibagi latih/uji DAN memakai cross validation?**
- CV (5-fold, di data latih) dipakai untuk **mengambil keputusan**, misalnya memilih α dan model aktif. CV memberi rata-rata ± std, jadi kita tahu seberapa stabil skornya.
- Data uji (20%) disimpan sampai akhir dan hanya dipakai sekali untuk **melaporkan** performa pada data yang benar-benar belum pernah dilihat.
- Jika data uji dipakai untuk memilih, skornya tidak lagi jujur.

**13. Kenapa RMSE uji (433) lebih besar dari RMSE CV (360)?**
- RMSE sangat sensitif terhadap beberapa error besar, dan data uji kebetulan berisi beberapa laptop sangat mahal. Razer Blade Pro €6.099 sendiri menyumbang error sekitar €3.165.
- Variasi antar-fold memang besar: RMSE per fold berkisar sekitar 296–445.
- MAPE uji (19,9%) justru sedikit lebih baik dari MAPE CV (20,6%), jadi model tidak memburuk secara umum.

**14. Apa arti MAE, RMSE, MAPE, R²?**
- **MAE**: rata-rata meleset dalam euro (€245).
- **RMSE**: seperti MAE, tetapi error besar dihukum lebih berat. RMSE ≫ MAE menandakan ada beberapa error ekstrem.
- **MAPE**: rata-rata meleset dalam persen (19,9%).
- **R²**: proporsi variasi yang dijelaskan. 0 = sama dengan menebak rata-rata, 1 = sempurna.

**15. Kenapa R² skala log (0,831) berbeda dengan R² skala euro (0,688)?**
Model mengoptimalkan error di skala log. Saat dikembalikan dengan exp(), error persentase yang sama menjadi error euro yang jauh lebih besar pada laptop mahal. Laptop mahal itu mendominasi jumlah kuadrat error di skala euro, sehingga R² euro lebih rendah. Keduanya kami laporkan.

**16. Bagaimana membuktikan bahwa implementasi manual benar?**
Ada tiga lapis:
1. Tes unit dengan data sintetis yang bobot aslinya diketahui.
2. Koefisien manual dibandingkan dengan scikit-learn: selisih 10⁻¹³ (Normal Equation) dan 10⁻¹⁴ (Ridge).
3. Metrik manual dibandingkan dengan `sklearn.metrics`: selisih 0.

**17. Apa fungsi baseline?**
Baseline adalah batas bawah. Model yang selalu menebak rata-rata harga mendapat MAPE uji 67,96% dan R² ≈ 0. Model kita (MAPE 19,89%) baru berarti jika jauh lebih baik dari baseline, dan memang lebih baik: MAPE turun 70,7%.

**18. Apa itu aturan 1-SE dan kenapa tidak langsung memilih RMSE CV terkecil?**
- Selisih kecil antar-model bisa saja hanya kebetulan pembagian fold.
- Aturan 1-SE: model dengan RMSE CV ≤ (RMSE terkecil + 1 standard error) dianggap setara, lalu dipilih yang paling sederhana.
- Polinomial (351,5) dan linear (359,8) berselisih 8 euro, jauh di bawah SE 27,6, sehingga dipilih linear: lebih mudah ditafsirkan dan lebih kecil risiko overfitting-nya.
- Sumber: Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, §7.10.

**19. Apakah model overfitting atau underfitting?**
- **Tidak overfitting:** RMSE latih 348 vs CV 360 (celah sekitar 3%). Di learning curve, kedua kurva berdekatan di akhir.
- **Agak underfitting:** kurva validasi sudah mendatar di sekitar 360 euro, jadi batasnya ada pada informasi fitur, bukan pada jumlah data.

**20. Apakah asumsi regresi linear terpenuhi?**
- **Linearitas & homoskedastisitas:** grafik residual vs prediksi acak di sekitar 0 tanpa pola corong; korelasi |residual|–prediksi 0,11.
- **Normalitas residual:** Q-Q plot hampir lurus, dengan sedikit ekor tebal di kanan. Shapiro-Wilk p = 0,090 > 0,05, jadi H₀ (normal) tidak ditolak.
- **Multikolinearitas:** VIF semua fitur numerik yang dipakai ≤ 3,8.

**21. Bagaimana rentang perkiraan harga dihitung dan apakah bisa dipercaya?**
- Rentangnya exp(ŷ ± 1,96·RMSE_log), dengan RMSE_log dari CV. Jika residual log normal, sekitar 95% harga aktual seharusnya ada di dalamnya.
- Kami **mengujinya** di data uji: 94,1% memang berada di dalam rentang, jadi rentang ini terkalibrasi.

**22. Bagaimana kontribusi fitur dihitung?**
- Modelnya linear: ln(harga) = w₀ + Σ wⱼxⱼ, sehingga sumbangan fitur j adalah wⱼxⱼ. Kolom one-hot dari fitur yang sama dijumlahkan.
- Titik acuannya w₀ = harga "laptop acuan" (semua fitur numerik rata-rata, kategori acuan).
- Tes menjamin intersep + jumlah kontribusi = prediksi ln(harga).

**23. Apa itu multikolinearitas dan bagaimana mendeteksinya?**
- Multikolinearitas terjadi jika fitur saling menjelaskan, sehingga bobotnya tidak stabil.
- Deteksinya dengan **VIF** = 1/(1 − R²ⱼ). VIF lebar, tinggi, dan PPI sekitar 125–130 jika dipakai bersama.
- Ukuran lain adalah condition number XᵀX: tak hingga (singular) pada data mentah, 4.119 dengan lebar/tinggi, dan 748 pada model akhir.
- Solusinya membuang fitur yang berlebihan, atau memakai Ridge.

### C. Data & feature engineering

**24. Kenapa kolom `Product` tidak dipakai?**
Ada 617 nilai unik untuk 1.274 laptop. Jika di-one-hot, model hanya akan menghafal nama produk (overfitting) dan tidak bisa memprediksi produk baru.

**25. Kenapa satu laptop dibuang?**
Itu satu-satunya laptop berprosesor ARM (Samsung Chromebook Plus). Koefisien tidak bisa dipelajari dari satu data, dan kolom CPU=Lainnya serta GPU=Lainnya menjadi identik (kolinearitas sempurna). Model dinyatakan berlaku untuk laptop Intel/AMD.

**26. Kenapa RAM memakai log2?**
Efek RAM terhadap harga berupa "kelipatan": 4→8 GB jauh lebih berarti daripada 32→36 GB. Dengan log2, koefisiennya berarti "+28% setiap RAM 2× lipat". Keputusan ini didukung CV (RMSE 389 → 360) dan mencegah ekstrapolasi liar untuk RAM 64 GB.

**27. Bagaimana menangani format memori "128GB SSD + 1TB HDD"?**
Teks dipecah pada tanda "+", lalu setiap bagian dibaca dengan regex `angka(GB|TB) jenis`. TB dikali 1000, dan hasilnya dijumlahkan per jenis. Semua 39 variasi format di dataset diuji di `test_preprocessing.py`.

**28. Bagaimana harga euro 2018 diubah ke rupiah sekarang?**
Rumusnya harga × (HICP Okt 2026 ÷ rata-rata HICP 2018) × kurs = harga × 1,302 × 20.010. Datanya bersumber dari Eurostat (via FRED) dan XE, dan tercatat di `penyesuaian_harga.json`.

Ini menyetarakan nilai uang, bukan harga pasar laptop baru. Faktor ini tidak mengubah MAPE/R², dan validasi dilakukan pada harga asli.

**29. Bagaimana jika input berada di luar data latih?**
API tetap memprediksi, tetapi memberi **peringatan**. Contohnya: "RAM 64 GB di luar rentang data latih (2–32 GB); prediksi merupakan ekstrapolasi…", atau "Razer … diperlakukan sebagai kelompok Lainnya". Input yang tidak masuk akal (RAM negatif, merek tidak dikenal, tanpa penyimpanan) ditolak dengan status 422.

### D. Sistem & implementasi

**30. Kenapa hasil evaluasi disimpan ke database, bukan dihitung ulang setiap kali?**
- **Satu sumber kebenaran:** angka di website = angka hasil training, tidak ada angka manual.
- **Versi:** setiap training menjadi versi baru, dan versi lama tetap bisa dibandingkan lewat dropdown.
- **Kecepatan:** website tidak perlu menjalankan CV setiap kali dibuka.

**31. Kenapa ada tabel `training_runs` selain 3 tabel minimal?**
Sebagian hasil berlaku untuk seluruh run, bukan per model: info dataset, pemilihan α, verifikasi koefisien antar-model, multikolinearitas, dan alasan pemilihan model. Tanpa tabel ini, data tersebut harus disalin ke ketujuh model.

**32. Bagaimana hasil bisa diulang (reproducible)?**
- `random_state = 42` untuk split dan K-Fold.
- Versi library dikunci di `requirements.txt`.
- Dua kali training menghasilkan CV, metrik, dan koefisien yang identik.
- Training di Docker (Linux, Python 3.12) dan di laptop (Windows, Python 3.14) juga menghasilkan angka yang sama.

**33. Apa yang dilakukan backend saat Docker dinyalakan?**
`siapkan.py` menunggu PostgreSQL, menjalankan migrasi Alembic, lalu melatih model hanya jika belum ada model aktif. Setelah itu uvicorn menyala. Saat restart, model tidak dilatih ulang.

**34. Pengguna awam tidak tahu clock CPU atau berat laptopnya. Bagaimana form menanganinya?**
- **CPU:** pengguna memilih **nama model prosesor** seperti yang tertulis di spesifikasi (mis. "Intel Core i5 7200U"). Dari dataset diketahui model itu termasuk kategori Intel Core i5 dan clock-nya 2,5 GHz, jadi keduanya terisi otomatis. Dari 92 model CPU, 80 hanya punya satu nilai clock; untuk 12 sisanya dipakai nilai yang paling sering, dan pilihan clock lain tetap tersedia.
- **GPU:** dipilih berdasarkan nama model (mis. "GeForce GTX 1050"), lalu merek GPU terisi otomatis. Model sendiri hanya memakai merek GPU.
- **Berat:** diisi otomatis dengan median berat laptop sejenis (tipe & ukuran layar yang sama), dan tetap bisa diubah.
- **Paling mudah:** cari laptop yang mirip di dataset, klik *Pakai*, lalu ubah bagian yang berbeda. Jika laptop tidak diubah, hasil prediksi dibandingkan langsung dengan harga aslinya. Laptop bertanda "data uji" memberi perbandingan yang jujur, karena tidak dipakai saat melatih model.

**35. Bagaimana meningkatkan model ke depan?**
- Menambah fitur yang lebih informatif: tipe GPU (GTX 1050 vs 1070), generasi CPU, seri premium.
- Memakai data harga terbaru agar benar-benar mencerminkan pasar sekarang.
- Mencoba model non-linear (pohon keputusan / gradient boosting) sebagai pembanding, tetap dengan baseline dan CV yang sama.
