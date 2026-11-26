"""
layanan_dataset.py — Ringkasan, data EDA, dan pilihan form, dihitung dari CSV.

Semua angka di halaman "Eksplorasi Data" dan pilihan dropdown di form
prediksi berasal dari fungsi-fungsi di sini, yang membaca CSV asli lewat
fungsi preprocessing yang SAMA dengan yang dipakai training. Hasilnya
dihitung sekali saat API dinyalakan lalu disimpan di memori (cache).
"""

from __future__ import annotations

import re
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


# Nama umum resolusi layar, agar pengguna tidak perlu hafal angka piksel
NAMA_RESOLUSI = {
    (1366, 768): "HD",
    (1440, 900): "WXGA+",
    (1600, 900): "HD+",
    (1920, 1080): "Full HD",
    (1920, 1200): "WUXGA",
    (2560, 1440): "QHD",
    (2560, 1600): "WQXGA",
    (3200, 1800): "QHD+",
    (3840, 2160): "4K UHD",
}
URUTAN_KATEGORI_CPU = ["Intel Core i7", "Intel Core i5", "Intel Core i3", "Intel Lainnya", "AMD"]
URUTAN_MEREK_GPU = ["Intel", "Nvidia", "AMD"]


def _label_cpu(perusahaan: str, tipe: str) -> str:
    """'Intel' + 'Core i5 7200U' -> 'Intel Core i5 7200U'.

    Beberapa baris hanya menulis seri tanpa nomor model (mis. 'Core i5',
    'Core M m3'); labelnya diberi keterangan agar pengguna tidak salah paham.
    Nomor model dikenali dari adanya minimal dua digit berurutan
    (7200U, N3350, 6Y75, 9420, ...).
    """
    label = f"{perusahaan} {tipe}"
    return label if re.search(r"\d{2}", tipe) else f"{label} (model tidak disebut)"


def daftar_model_cpu(data: DataDataset) -> list[dict]:
    """Setiap model prosesor di dataset beserta kategori & clock-nya.

    Pengguna cukup memilih model CPU yang tertulis di spesifikasi laptopnya;
    kategori (i3/i5/i7/...) dan clock GHz diisi otomatis. Clock = nilai yang
    paling sering muncul untuk model itu (80 dari 92 model hanya punya satu
    nilai clock di dataset).
    """
    tabel = data.bersih.assign(CPU_Kategori=data.fitur["CPU_Kategori"].to_numpy())
    hasil = []
    for (perusahaan, tipe), grup in tabel.groupby(["CPU_Company", "CPU_Type"]):
        ghz = grup["CPU_GHz"].astype(float)
        hasil.append(
            {
                "nilai": _label_cpu(perusahaan, tipe),
                "kategori": str(grup["CPU_Kategori"].iloc[0]),
                "ghz": float(ghz.mode().iloc[0]),
                "ghz_variasi": sorted(float(v) for v in ghz.unique()),
                "jumlah": int(len(grup)),
            }
        )
    hasil.sort(key=lambda d: (URUTAN_KATEGORI_CPU.index(d["kategori"]), -d["jumlah"], d["nilai"]))
    return hasil


def daftar_model_gpu(data: DataDataset) -> list[dict]:
    """Setiap model kartu grafis di dataset. Model hanya memakai MEREK GPU-nya,
    tetapi pengguna lebih mudah mengenali nama modelnya (mis. GeForce GTX 1050)."""
    tabel = data.bersih.assign(GPU_Merek=data.fitur["GPU_Merek"].to_numpy())
    hasil = [
        {"nilai": f"{perusahaan} {tipe}", "merek": str(grup["GPU_Merek"].iloc[0]), "jumlah": int(len(grup))}
        for (perusahaan, tipe), grup in tabel.groupby(["GPU_Company", "GPU_Type"])
    ]
    hasil.sort(key=lambda d: (URUTAN_MEREK_GPU.index(d["merek"]), -d["jumlah"], d["nilai"]))
    return hasil


def berat_tipikal(data: DataDataset) -> dict:
    """Median berat per (tipe laptop, ukuran layar), dengan cadangan per tipe.

    Dipakai form untuk mengisi berat otomatis, karena kebanyakan orang tidak
    hafal berat laptopnya. Pengguna tetap bisa mengubahnya.
    """
    F = data.fitur
    per_tipe_inci = F.groupby(["TypeName", "Inches"])["Weight"].median()
    return {
        "per_tipe_inci": {f"{t}|{float(i):g}": round(float(v), 2) for (t, i), v in per_tipe_inci.items()},
        "per_tipe": {str(t): round(float(v), 2) for t, v in F.groupby("TypeName")["Weight"].median().items()},
        "semua": round(float(F["Weight"].median()), 2),
    }


def opsi_form(data: DataDataset) -> dict:
    """Isi endpoint GET /api/options: pilihan dropdown & batas nilai input.

    Pilihan kategori diurutkan dari yang paling sering muncul. Nilai default
    form = konfigurasi yang paling umum di dataset.
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
    cpu = daftar_model_cpu(data)
    gpu = daftar_model_gpu(data)
    berat = berat_tipikal(data)
    cpu_umum = max(cpu, key=lambda d: d["jumlah"])
    gpu_umum = max(gpu, key=lambda d: d["jumlah"])
    tipe, inci = str(modus["TypeName"]), float(modus["Inches"])
    return {
        "company": kategori("Company"),
        "type_name": kategori("TypeName"),
        "cpu_kategori": kategori("CPU_Kategori"),
        "cpu_model": cpu,
        "gpu_merek": kategori("GPU_Merek"),
        "gpu_model": gpu,
        "os": kategori("OS"),
        "resolusi": [
            {
                "lebar_px": int(r.Lebar_px),
                "tinggi_px": int(r.Tinggi_px),
                "nama": NAMA_RESOLUSI.get((int(r.Lebar_px), int(r.Tinggi_px))),
                "jumlah": int(r.jumlah),
            }
            for r in resolusi.itertuples()
        ],
        "ram_gb": unik_terurut("Ram"),
        "ssd_gb": unik_terurut("SSD_GB"),
        "hdd_gb": unik_terurut("HDD_GB"),
        "flash_gb": unik_terurut("Flash_GB"),
        "inches": unik_terurut("Inches"),
        "berat_tipikal": berat,
        "rentang": {
            k: {"min": float(F[k].min()), "max": float(F[k].max())}
            for k in ["Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px", "SSD_GB", "HDD_GB", "Flash_GB"]
        },
        "nilai_default": {
            "company": str(modus["Company"]),
            "type_name": tipe,
            "inches": inci,
            "ram_gb": float(modus["Ram"]),
            "berat_kg": berat["per_tipe_inci"].get(f"{tipe}|{inci:g}", berat["semua"]),
            "cpu_model": cpu_umum["nilai"],
            "cpu_kategori": cpu_umum["kategori"],
            "cpu_ghz": cpu_umum["ghz"],
            "lebar_px": int(resolusi.iloc[0].Lebar_px),
            "tinggi_px": int(resolusi.iloc[0].Tinggi_px),
            "touchscreen": False,
            "ips": False,
            "ssd_gb": float(modus["SSD_GB"]),
            "hdd_gb": float(modus["HDD_GB"]),
            "flash_gb": 0.0,
            "gpu_model": gpu_umum["nilai"],
            "gpu_merek": gpu_umum["merek"],
            "os": str(modus["OS"]),
        },
    }


# ---------------------------------------------------------------------------
# Daftar laptop di dataset (untuk "isi dari laptop yang sudah ada")
# ---------------------------------------------------------------------------
def daftar_laptop(data: DataDataset) -> list[dict]:
    """Semua laptop di dataset dalam bentuk input form, beserta harga aslinya.

    Setiap laptop diberi tanda "latih" atau "uji" sesuai pembagian data yang
    SAMA dengan training (fungsi bagi_latih_uji, random_state tetap). Laptop
    bertanda "uji" belum pernah dilihat model, sehingga membandingkan
    prediksinya dengan harga asli adalah uji yang jujur.
    """
    from ml.evaluation import bagi_latih_uji

    df, F = data.bersih, data.fitur
    _, idx_uji = bagi_latih_uji(len(df))
    uji = set(int(i) for i in idx_uji)
    hasil = []
    for i in range(len(df)):
        b, f = df.iloc[i], F.iloc[i]
        cpu = _label_cpu(b["CPU_Company"], b["CPU_Type"])
        gpu = f"{b['GPU_Company']} {b['GPU_Type']}"
        hasil.append(
            {
                "id": i,
                "label": f"{b['Company']} {b['Product']}",
                "ringkas": f"{b['TypeName']} · {cpu} · RAM {int(f['Ram'])} GB · {b['Memory']} · {float(f['Inches']):g} inci",
                "harga_euro": float(b["Price_euros"]),
                "set": "uji" if i in uji else "latih",
                "input": {
                    "company": str(f["Company"]),
                    "type_name": str(f["TypeName"]),
                    "inches": float(f["Inches"]),
                    "ram_gb": float(f["Ram"]),
                    "berat_kg": float(f["Weight"]),
                    "cpu_model": cpu,
                    "cpu_kategori": str(f["CPU_Kategori"]),
                    "cpu_ghz": float(f["CPU_GHz"]),
                    "lebar_px": int(f["Lebar_px"]),
                    "tinggi_px": int(f["Tinggi_px"]),
                    "touchscreen": bool(f["Touchscreen"]),
                    "ips": bool(f["IPS"]),
                    "ssd_gb": float(f["SSD_GB"]),
                    "hdd_gb": float(f["HDD_GB"]),
                    "flash_gb": float(f["Flash_GB"]),
                    "gpu_model": gpu,
                    "gpu_merek": str(f["GPU_Merek"]),
                    "os": str(f["OS"]),
                },
                "_teks": f"{b['Company']} {b['Product']} {b['TypeName']} {cpu} {gpu} {int(f['Ram'])}gb".lower(),
            }
        )
    return hasil


def cari_laptop(daftar: list[dict], kata_kunci: str, batas: int = 12) -> dict:
    """Mencari laptop yang mengandung SEMUA kata kunci (mis. "dell i7 gaming").

    Laptop dari data UJI ditampilkan lebih dulu, karena hanya laptop itulah
    yang memberi perbandingan "prediksi vs harga asli" yang jujur (belum
    pernah dilihat model). Tanpa kata kunci, dikembalikan contoh laptop dari
    data uji yang tersebar dari yang termurah sampai termahal.
    """
    kata = kata_kunci.lower().split()
    if kata:
        cocok = [d for d in daftar if all(k in d["_teks"] for k in kata)]
        cocok.sort(key=lambda d: d["set"] != "uji")  # sort stabil: uji dulu, urutan lain tetap
    else:
        uji = sorted((d for d in daftar if d["set"] == "uji"), key=lambda d: d["harga_euro"])
        langkah = max(1, len(uji) // batas)
        cocok = uji[::langkah]
    return {
        "total_cocok": len(cocok),
        "data": [{k: v for k, v in d.items() if k != "_teks"} for d in cocok[:batas]],
    }
