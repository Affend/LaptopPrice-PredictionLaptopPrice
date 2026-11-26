"""
layanan_dataset.py — Ringkasan, data EDA, dan pilihan form, dihitung dari CSV.

Semua angka di halaman "Eksplorasi Data" dan pilihan dropdown di form
prediksi berasal dari fungsi-fungsi di sini, yang membaca CSV asli lewat
fungsi preprocessing yang SAMA dengan yang dipakai training. Hasilnya
dihitung sekali saat API dinyalakan lalu disimpan di memori (cache).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ml.preprocessing import bersihkan_data, ekstrak_fitur, muat_dataset

# Nama tampilan kolom numerik (satuan asli)
LABEL_NUMERIK = {
    "Price_euros": "Harga (euro)",
    "Inches": "Ukuran layar (inci)",
    "Ram": "RAM (GB)",
    "Weight": "Berat (kg)",
    "CPU_GHz": "Frekuensi CPU (GHz)",
    "Lebar_px": "Lebar resolusi (piksel)",
    "Tinggi_px": "Tinggi resolusi (piksel)",
    "PPI": "Kerapatan piksel (PPI)",
    "SSD_GB": "SSD (GB)",
    "HDD_GB": "HDD (GB)",
    "Flash_GB": "Flash storage (GB)",
}


@dataclass
class DataDataset:
    """Data yang dimuat sekali saat API dinyalakan."""

    mentah: pd.DataFrame      # CSV apa adanya (nama kolom sudah diseragamkan)
    bersih: pd.DataFrame      # setelah baris di luar cakupan dibuang
    fitur: pd.DataFrame       # hasil ekstrak_fitur()
    dibuang: list[dict]
    path: Path


def muat_data_dataset(path: Path) -> DataDataset:
    mentah = muat_dataset(path)
    bersih, dibuang = bersihkan_data(mentah)
    return DataDataset(mentah=mentah, bersih=bersih, fitur=ekstrak_fitur(bersih), dibuang=dibuang, path=Path(path))


def _statistik(nilai: pd.Series) -> dict:
    nilai = nilai.astype(float)
    return {
        "jumlah": int(nilai.count()),
        "rata_rata": float(nilai.mean()),
        "std": float(nilai.std()),
        "min": float(nilai.min()),
        "q1": float(nilai.quantile(0.25)),
        "median": float(nilai.median()),
        "q3": float(nilai.quantile(0.75)),
        "max": float(nilai.max()),
        "skewness": float(nilai.skew()),
    }


def ringkasan(data: DataDataset) -> dict:
    """Isi endpoint GET /api/dataset/summary."""
    tabel = data.fitur.assign(Price_euros=data.bersih["Price_euros"].to_numpy())
    return {
        "nama_file": data.path.name,
        "jumlah_baris_mentah": int(len(data.mentah)),
        "jumlah_kolom_mentah": int(data.mentah.shape[1]),
        "kolom_mentah": list(data.mentah.columns),
        "jumlah_baris_dipakai": int(len(data.bersih)),
        "baris_dibuang": data.dibuang,
        "nilai_kosong": int(data.mentah.isna().sum().sum()),
        "baris_duplikat": int(data.mentah.duplicated().sum()),
        "jumlah_produk_unik": int(data.bersih["Product"].nunique()),
        "statistik_numerik": [
            {"kolom": k, "label": LABEL_NUMERIK[k], **_statistik(tabel[k])} for k in LABEL_NUMERIK
        ],
        "statistik_log_harga": _statistik(np.log(tabel["Price_euros"])),
        "jumlah_per_kategori": {
            kolom: [{"kategori": str(k), "jumlah": int(v)} for k, v in data.fitur[kolom].value_counts().items()]
            for kolom in ["Company", "TypeName", "CPU_Kategori", "GPU_Merek", "OS"]
        },
        "contoh_baris": data.mentah.head(5).to_dict(orient="records"),
    }


def _histogram(nilai: np.ndarray, jumlah_bin: int = 30) -> list[dict]:
    frekuensi, tepi = np.histogram(nilai, bins=jumlah_bin)
    return [
        {"bin_kiri": float(tepi[i]), "bin_kanan": float(tepi[i + 1]), "tengah": float((tepi[i] + tepi[i + 1]) / 2),
         "frekuensi": int(frekuensi[i])}
        for i in range(len(frekuensi))
    ]


def _harga_per(tabel: pd.DataFrame, kolom: str) -> list[dict]:
    """Statistik harga per kelompok (untuk grafik batang & ringkasan box plot)."""
    grup = tabel.groupby(kolom)["Price_euros"]
    hasil = pd.DataFrame(
        {
            "jumlah": grup.size(),
            "rata_rata": grup.mean(),
            "median": grup.median(),
            "q1": grup.quantile(0.25),
            "q3": grup.quantile(0.75),
            "min": grup.min(),
            "max": grup.max(),
        }
    ).sort_values("median", ascending=False)
    return [{"grup": str(idx), **{k: float(v) for k, v in baris.items()}} for idx, baris in hasil.iterrows()]


def eda(data: DataDataset) -> dict:
    """Isi endpoint GET /api/dataset/eda."""
    tabel = data.fitur.assign(Price_euros=data.bersih["Price_euros"].to_numpy())
    harga = tabel["Price_euros"].to_numpy(float)
    kolom_korelasi = ["Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px", "PPI", "SSD_GB", "HDD_GB", "Flash_GB"]
    matriks = tabel[kolom_korelasi].assign(ln_harga=np.log(harga)).corr()
    tabel["Ram_kategori"] = tabel["Ram"].astype(int).astype(str) + " GB"

    return {
        "distribusi_harga": _histogram(harga),
        "distribusi_log_harga": _histogram(np.log(harga)),
        "skewness": {"harga": float(pd.Series(harga).skew()), "log_harga": float(pd.Series(np.log(harga)).skew())},
        "korelasi": {
            "kolom": list(matriks.columns),
            "nilai": [[float(v) for v in baris] for baris in matriks.to_numpy()],
        },
        "korelasi_dengan_log_harga": sorted(
            [{"fitur": k, "label": LABEL_NUMERIK[k], "korelasi": float(matriks.at[k, "ln_harga"])} for k in kolom_korelasi],
            key=lambda d: -abs(d["korelasi"]),
        ),
        "harga_per_merek": _harga_per(tabel, "Company"),
        "harga_per_tipe": _harga_per(tabel, "TypeName"),
        "harga_per_cpu": _harga_per(tabel, "CPU_Kategori"),
        "harga_per_gpu": _harga_per(tabel, "GPU_Merek"),
        "harga_per_os": _harga_per(tabel, "OS"),
        "harga_per_ram": sorted(_harga_per(tabel, "Ram_kategori"), key=lambda d: float(d["grup"].split()[0])),
        # titik sebar RAM/PPI vs harga (semua 1274 baris, ringan)
        "sebaran": [
            {"ram": float(r), "ppi": round(float(p), 1), "ssd": float(s), "berat": float(w), "harga": float(h),
             "tipe": str(t), "merek": str(c)}
            for r, p, s, w, h, t, c in zip(tabel["Ram"], tabel["PPI"], tabel["SSD_GB"], tabel["Weight"],
                                           harga, tabel["TypeName"], tabel["Company"])
        ],
    }


def opsi_form(data: DataDataset) -> dict:
    """Isi endpoint GET /api/options: pilihan dropdown & batas nilai input.

    Pilihan kategori diurutkan dari yang paling sering muncul. Nilai default
    form = nilai yang paling umum (modus) di dataset.
    """
    F = data.fitur

    def kategori(kolom):
        return [{"nilai": str(k), "jumlah": int(v)} for k, v in F[kolom].value_counts().items()]

    def unik_terurut(kolom):
        return sorted(float(v) for v in F[kolom].unique())

    resolusi = (
        F.groupby(["Lebar_px", "Tinggi_px"]).size().reset_index(name="jumlah").sort_values("jumlah", ascending=False)
    )
    modus = F.mode().iloc[0]
    return {
        "company": kategori("Company"),
        "type_name": kategori("TypeName"),
        "cpu_kategori": kategori("CPU_Kategori"),
        "gpu_merek": kategori("GPU_Merek"),
        "os": kategori("OS"),
        "resolusi": [
            {"lebar_px": int(r.Lebar_px), "tinggi_px": int(r.Tinggi_px), "jumlah": int(r.jumlah)}
            for r in resolusi.itertuples()
        ],
        "ram_gb": unik_terurut("Ram"),
        "ssd_gb": unik_terurut("SSD_GB"),
        "hdd_gb": unik_terurut("HDD_GB"),
        "flash_gb": unik_terurut("Flash_GB"),
        "inches": unik_terurut("Inches"),
        "rentang": {
            k: {"min": float(F[k].min()), "max": float(F[k].max())}
            for k in ["Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px", "SSD_GB", "HDD_GB", "Flash_GB"]
        },
        "nilai_default": {
            "company": str(modus["Company"]),
            "type_name": str(modus["TypeName"]),
            "inches": float(modus["Inches"]),
            "ram_gb": float(modus["Ram"]),
            "berat_kg": float(F["Weight"].median()),
            "cpu_kategori": str(modus["CPU_Kategori"]),
            "cpu_ghz": float(F["CPU_GHz"].median()),
            "lebar_px": int(resolusi.iloc[0].Lebar_px),
            "tinggi_px": int(resolusi.iloc[0].Tinggi_px),
            "touchscreen": False,
            "ips": False,
            "ssd_gb": float(modus["SSD_GB"]),
            "hdd_gb": float(modus["HDD_GB"]),
            "flash_gb": 0.0,
            "gpu_merek": str(modus["GPU_Merek"]),
            "os": str(modus["OS"]),
        },
    }
