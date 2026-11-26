# LaptopPrice: Sistem Prediksi Harga Laptop Berdasarkan Spesifikasi Menggunakan Regresi Linear Berganda

Tugas Besar mata kuliah Kecerdasan Buatan: pengenalan pola berbasis pembelajaran mesin dengan metode **regresi**.

Sistem ini memprediksi harga laptop dari spesifikasinya (merek, tipe, CPU, RAM, penyimpanan, layar, GPU, OS, berat). Model utamanya adalah **regresi linear berganda yang diimplementasikan manual dengan NumPy** (Normal Equation dan Gradient Descent). Model ini dibandingkan dengan scikit-learn, Ridge, regresi polinomial, dan baseline, lalu divalidasi secara menyeluruh.

| Bagian | Teknologi |
|---|---|
| Pipeline ML | Python, NumPy, pandas, scikit-learn (hanya sebagai pembanding & verifikasi), SciPy, matplotlib |
| Backend API | FastAPI, Pydantic, SQLAlchemy 2, Alembic |
| Database | PostgreSQL (Docker) atau SQLite (tanpa Docker) |
| Frontend | Next.js 16 (App Router, TypeScript), Tailwind CSS, Recharts, KaTeX |
| Deploy lokal | Docker Compose |

Penjelasan lengkap setiap file, alur data, dan daftar pertanyaan dosen ada di **[PENJELASAN.md](PENJELASAN.md)**.

---

## Ringkasan hasil (training versi 1, 26 November 2026)

Model aktif: **Regresi Linear Berganda, Normal Equation (manual)**, dipilih dengan aturan 1-SE dari 5-Fold Cross Validation.

| | MAE | RMSE | MAPE | R² (euro) | R² (log) |
|---|---|---|---|---|---|
| 5-Fold CV (data latih, rata-rata ± std) | €236,4 | €359,8 ± 57,1 | 20,58% ± 1,11 | 0,718 | 0,820 |
| Data uji (255 laptop) | €244,7 | €433,4 | 19,89% | 0,688 | 0,831 |
| Baseline rata-rata (data uji) | €537,6 | €777,0 | 67,96% | −0,002 | −0,064 |

Angka di atas diambil dari `backend/outputs/hasil_evaluasi.json`. Angka terkini selalu bisa dilihat di halaman **Validasi & Evaluasi** pada website, yang membacanya dari database.

---

## Struktur proyek

```
LaptopPrice-PredictionLaptopPrice/
├── docker-compose.yml        # postgres + backend + frontend
├── .env.example              # contoh konfigurasi (salin menjadi .env)
├── README.md / PENJELASAN.md
├── backend/
│   ├── data/
│   │   ├── laptop_price.csv          # dataset Kaggle
│   │   └── penyesuaian_harga.json    # data inflasi (HICP) & kurs beserta sumbernya
│   ├── ml/                   # pipeline ML
│   │   ├── config.py         # pengaturan & konstanta eksperimen
│   │   ├── preprocessing.py  # parsing, feature engineering, Preprocessor
│   │   ├── model.py          # Normal Equation, Gradient Descent, Ridge, baseline (manual)
│   │   ├── evaluation.py     # metrik manual, CV, residual, VIF, dll.
│   │   ├── train.py          # menjalankan seluruh pipeline
│   │   └── plots.py          # ekspor gambar PNG untuk laporan
│   ├── db/                   # koneksi, tabel ORM, repository
│   ├── alembic/              # migrasi database
│   ├── app/                  # FastAPI (main.py, rute.py, skema.py, layanan_*.py)
│   ├── tests/                # pytest
│   ├── models/               # file model .joblib (dibuat saat training)
│   ├── outputs/              # gambar PNG + hasil_evaluasi.json (bahan laporan)
│   ├── analysis.ipynb        # notebook analisis (bahan laporan & presentasi)
│   ├── siapkan.py            # persiapan otomatis saat container menyala
│   └── Dockerfile
└── frontend/
    ├── app/                  # halaman: /, /eksplorasi, /prediksi, /evaluasi, /metode
    ├── components/           # komponen UI & grafik
    ├── lib/                  # klien API, format angka, tipe data, palet warna
    └── Dockerfile
```

---

## Prasyarat (Windows)

- **Cara 1 (Docker):** [Docker Desktop](https://www.docker.com/products/docker-desktop/) dalam kondisi menyala.
- **Cara 2 (tanpa Docker):** Python 3.11+ (diuji dengan 3.12 dan 3.14) dan Node.js LTS (diuji dengan Node 24).

Semua perintah di bawah ditulis untuk **PowerShell**, dijalankan dari folder root proyek.

---

## Cara 1: Docker Compose (disarankan)

1. (Opsional) Buat file konfigurasi. Tanpa `.env`, nilai default di `docker-compose.yml` yang dipakai.
   ```powershell
   Copy-Item .env.example .env
   ```
2. Build dan jalankan semua service:
   ```powershell
   docker compose up --build
   ```
   Saat pertama kali menyala, backend otomatis menunggu PostgreSQL siap, menjalankan migrasi Alembic, lalu **melatih model** (sekitar 10 detik) karena database masih kosong. Tunggu sampai muncul `Application startup complete`.
3. Buka:
   - Website: <http://localhost:3000>
   - Dokumentasi API interaktif (Swagger): <http://localhost:8000/docs>

Perintah lain yang berguna:

```powershell
docker compose ps
```
```powershell
docker compose logs -f backend
```
```powershell
docker compose exec postgres psql -U laptopprice -d laptopprice
```
```powershell
docker compose down
```

`docker compose down` mematikan container, tetapi data tetap tersimpan di volume (database, model, gambar). Untuk menghapus **semua** data dan memulai dari nol, tambahkan `-v`:

```powershell
docker compose down -v
```

---

## Cara 2: Tanpa Docker (SQLite)

Tanpa `DATABASE_URL`, backend otomatis memakai SQLite (file `backend/laptopprice.db`).

### Backend

```powershell
cd backend
```
```powershell
python -m venv .venv
```
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```
```powershell
.\.venv\Scripts\alembic.exe upgrade head
```
```powershell
.\.venv\Scripts\python.exe -m ml.train
```
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Keterangan:
- `alembic upgrade head` membuat tabel database (cukup sekali).
- `python -m ml.train` melatih ketujuh model, menyimpan hasil evaluasi ke database sebagai versi baru, dan mengekspor gambar PNG ke `backend/outputs/`.
- `uvicorn` menyalakan API di <http://localhost:8000>.
- Jika ingin memakai `.\.venv\Scripts\Activate.ps1` lalu muncul galat *execution policy*, jalankan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. Cara lain, panggil `.venv\Scripts\python.exe` langsung seperti di atas.
- `requirements-dev.txt` = `requirements.txt` + Jupyter untuk notebook.

### Frontend (terminal baru)

```powershell
cd frontend
```
```powershell
npm install
```
```powershell
npm run dev
```

Buka <http://localhost:3000>. Frontend memanggil API di `http://localhost:8000`. Jika alamatnya berbeda, buat file `frontend/.env.local` berisi `NEXT_PUBLIC_API_URL=http://alamat-api:port`, lalu jalankan ulang `npm run dev`.

---

## Melatih ulang model

Setiap kali `python -m ml.train` dijalankan, terbentuk **nomor versi baru** di database. Model terpilih ditandai aktif, dan semua versi lama tetap bisa dilihat lewat dropdown di halaman Validasi & Evaluasi. Model aktif dimuat sekali saat API menyala, jadi **nyalakan ulang backend** setelah training.

- Tanpa Docker: jalankan `.\.venv\Scripts\python.exe -m ml.train` di folder `backend`, lalu nyalakan ulang uvicorn.
- Dengan Docker:
  ```powershell
  docker compose exec backend python -m ml.train
  ```
  ```powershell
  docker compose restart backend
  ```

Untuk mencoba pipeline tanpa menyentuh database: `python -m ml.train --no-db` (model disimpan ke `backend/models/lokal/`).

---

## Pengujian

```powershell
cd backend
```
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Terdapat 68 tes:
- [test_metrics_and_models.py](backend/tests/test_metrics_and_models.py): metrik manual = `sklearn.metrics`, Normal Equation / Ridge manual = scikit-learn, Gradient Descent konvergen.
- [test_preprocessing.py](backend/tests/test_preprocessing.py): parsing setiap format kolom, tidak ada data leakage, fitur dari form = fitur dari CSV.
- [test_database.py](backend/tests/test_database.py): migrasi, versi model, hanya satu model aktif, JSON aman untuk PostgreSQL.
- [test_api.py](backend/tests/test_api.py): semua endpoint, terutama `/api/predict` (valid, input salah, ekstrapolasi, tanpa model).

Tes memakai database SQLite sementara, sehingga database asli tidak tersentuh. Di Docker: `docker compose exec backend python -m pytest -q -p no:cacheprovider`.

Frontend:

```powershell
cd frontend
```
```powershell
npm run lint
```
```powershell
npm run build
```

---

## Notebook analisis

```powershell
cd backend
```
```powershell
.\.venv\Scripts\jupyter.exe notebook analysis.ipynb
```

Notebook memanggil fungsi yang sama di `ml/`, sehingga angkanya sama dengan website. Notebook tidak menulis ke database. Gambar PNG untuk laporan ada di `backend/outputs/` (dibuat ulang setiap training).

---

## Endpoint API

| Metode | Endpoint | Isi |
|---|---|---|
| GET | `/api/health` | status API, database, dan model aktif |
| GET | `/api/dataset/summary` | ringkasan & statistik dataset |
| GET | `/api/dataset/eda` | data grafik eksplorasi (distribusi, korelasi, harga per kelompok) |
| GET | `/api/options` | pilihan dropdown form (dari dataset) + data kurs/inflasi |
| POST | `/api/predict` | prediksi harga (IDR & EUR), rentang, kontribusi fitur, peringatan; disimpan ke tabel `predictions` |
| GET | `/api/models` | daftar semua versi model & metriknya |
| GET | `/api/evaluation/{model_id}` | semua hasil evaluasi satu model (untuk grafik) |
| GET | `/api/predictions/history?batas=20` | riwayat prediksi |

---

## Kurs & penyesuaian harga

Harga di dataset berada pada tingkat harga **sekitar 2018** (tahun dasar ini disimpulkan dari generasi hardware-nya; Kaggle hanya mengunggahnya tahun 2024). Prediksi model dalam euro 2018 dikalikan:

- **faktor inflasi zona euro** = HICP Okt 2026 (104,77) ÷ rata-rata HICP 2018 (80,47) = **1,302**. Sumber: Eurostat via FRED seri CP0000EZ19M086NEST.
- **kurs** Rp20.010/€ (XE, 25 Nov 2026).

Semua angka beserta sumbernya ada di [backend/data/penyesuaian_harga.json](backend/data/penyesuaian_harga.json). Untuk menimpanya tanpa mengubah file, isi `KURS_EUR_IDR` dan/atau `FAKTOR_INFLASI` di `.env`, lalu nyalakan ulang backend.

> Penyesuaian inflasi menyetarakan **nilai uang**, bukan harga pasar laptop baru: laptop dengan spesifikasi setara 2018 hari ini umumnya dijual lebih murah.

---

## Masalah umum

| Gejala | Penyebab & solusi |
|---|---|
| Website: “Backend tidak dapat dihubungi” | API belum menyala. Jalankan uvicorn (Cara 2) atau `docker compose up` (Cara 1). |
| `/api/predict` menjawab 503 “Belum ada model aktif” | Jalankan `python -m ml.train`, lalu nyalakan ulang backend. |
| `failed to connect to the docker API` | Docker Desktop belum menyala. |
| `port is already allocated` / port 8000 atau 3000 terpakai | Matikan server lokal (uvicorn / `npm run dev`) sebelum `docker compose up`, atau sebaliknya. |
| Galat *execution policy* saat aktivasi venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, atau panggil `.venv\Scripts\python.exe` langsung. |
| Model baru tidak terpakai setelah training | Model dimuat saat API menyala. Nyalakan ulang backend. |

---

## Sumber data

- Dataset: Kaggle “Laptop Price” (file `laptop_price - dataset.csv`, diunggah 12 Nov 2024), disimpan sebagai `backend/data/laptop_price.csv`.
- Inflasi: Eurostat, *Harmonized Index of Consumer Prices: Total for Euro Area (19 Countries)*, via [FRED CP0000EZ19M086NEST](https://fred.stlouisfed.org/series/CP0000EZ19M086NEST).
- Kurs: [XE mid-market rate](https://www.xe.com/currencyconverter/convert/?Amount=1&From=EUR&To=IDR), 25 November 2026.
