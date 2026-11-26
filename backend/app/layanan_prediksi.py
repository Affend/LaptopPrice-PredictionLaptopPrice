"""
layanan_prediksi.py — Memuat model aktif dan menghitung prediksi harga.

Alur satu prediksi (dipanggil oleh POST /api/predict):
    input form (sudah divalidasi Pydantic)
      -> fitur_dari_input()           : hitung PPI & fitur log2 (rumus sama dgn training)
      -> model.predict_log()          : y_hat = ln(harga) pada tingkat harga dataset (2018)
      -> exp(y_hat)                   : harga dalam euro (tingkat harga dataset)
      -> exp(y_hat ± 1,96·RMSE_log)   : rentang perkiraan
      -> × faktor inflasi             : setara euro sekarang
      -> × kurs EUR→IDR               : rupiah sekarang
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sqlalchemy.orm import Session

from db import repository
from ml.config import Z_95
from ml.model import ModelHargaLaptop
from ml.preprocessing import KATEGORI_LAINNYA, fitur_dari_input

# Nama tampilan untuk grafik kontribusi fitur
LABEL_FITUR = {
    "Company": "Merek",
    "TypeName": "Tipe laptop",
    "Inches": "Ukuran layar",
    "Ram_log2": "RAM",
    "Weight": "Berat",
    "CPU_GHz": "Frekuensi CPU",
    "PPI": "Kerapatan piksel (PPI)",
    "SSD_log2": "SSD",
    "HDD_log2": "HDD",
    "Flash_log2": "Flash storage",
    "Touchscreen": "Layar sentuh",
    "IPS": "Panel IPS",
    "CPU_Kategori": "Kategori CPU",
    "GPU_Merek": "Merek GPU",
    "OS": "Sistem operasi",
    "Polinomial": "Suku polinomial (kuadrat & interaksi)",
}
LABEL_SATUAN_ASLI = {
    "Inches": ("Ukuran layar", "inci"),
    "Ram": ("RAM", "GB"),
    "Weight": ("Berat", "kg"),
    "CPU_GHz": ("Frekuensi CPU", "GHz"),
    "Lebar_px": ("Lebar resolusi", "piksel"),
    "Tinggi_px": ("Tinggi resolusi", "piksel"),
    "PPI": ("Kerapatan piksel", "PPI"),
    "SSD_GB": ("SSD", "GB"),
    "HDD_GB": ("HDD", "GB"),
    "Flash_GB": ("Flash storage", "GB"),
}


@dataclass
class ModelAktif:
    """Model yang sedang dipakai API, beserta identitasnya di database."""

    id: int
    nomor_versi: int
    nama_model: str
    label: str
    dibuat_pada: str | None
    rmse_log_cv: float
    model: ModelHargaLaptop

    def info(self) -> dict:
        return {
            "id": self.id,
            "nomor_versi": self.nomor_versi,
            "nama_model": self.nama_model,
            "label": self.label,
            "dibuat_pada": self.dibuat_pada,
            "rmse_log_cv": self.rmse_log_cv,
        }


def muat_model_aktif(session: Session) -> ModelAktif | None:
    """Membaca baris model aktif dari database lalu memuat file .joblib-nya."""
    versi = repository.ambil_model_aktif(session)
    if versi is None:
        return None
    paket = joblib.load(repository.path_absolut(versi.path_file))
    return ModelAktif(
        id=versi.id,
        nomor_versi=versi.training_run_id,
        nama_model=versi.nama_model,
        label=versi.label,
        dibuat_pada=versi.dibuat_pada.isoformat() if versi.dibuat_pada else None,
        rmse_log_cv=float(paket["rmse_log_cv"]),
        model=paket["model"],
    )


def _peringatan(model: ModelHargaLaptop, data: dict) -> list[str]:
    """Catatan jika input berada di luar pengalaman data latih model."""
    pesan = []
    pp = model.preprocessor
    for kolom, (bawah, atas) in getattr(pp, "rentang_latih_", {}).items():
        if kolom not in data:
            continue
        nilai = float(data[kolom])
        if nilai < bawah or nilai > atas:
            nama, satuan = LABEL_SATUAN_ASLI.get(kolom, (kolom, ""))
            pesan.append(
                f"{nama} {nilai:g} {satuan} di luar rentang data latih ({bawah:g}–{atas:g} {satuan}); "
                "prediksi merupakan ekstrapolasi dan kurang dapat dipercaya."
            )
    for kolom in pp.fitur_kategori:
        nilai = str(data[kolom])
        if nilai not in pp.kategori_[kolom]:
            if KATEGORI_LAINNYA in pp.kategori_[kolom]:
                perlakuan = f"kelompok '{KATEGORI_LAINNYA}'"
            else:
                perlakuan = f"kategori acuan '{pp.kategori_acuan_[kolom]}'"
            pesan.append(
                f"'{nilai}' jarang muncul di data latih (kurang dari {pp.min_frekuensi} data), "
                f"sehingga model memperlakukannya sebagai {perlakuan}."
            )
    for duplikat in pp.kolom_duplikat_:
        fitur, kategori = duplikat["dibuang"].split("=", 1)
        if str(data.get(fitur)) == kategori:
            pesan.append(
                f"{fitur} = {kategori} selalu muncul bersama {duplikat['identik_dengan']} di data latih, "
                f"sehingga efeknya sudah termasuk dalam {duplikat['identik_dengan']}."
            )
    return pesan


def hitung_prediksi(aktif: ModelAktif, data: dict, penyesuaian: dict) -> dict:
    """Menghitung harga, rentang, kontribusi fitur, dan peringatan.

    `data` memakai nama kolom internal (Company, Ram, SSD_GB, ...).
    `penyesuaian` berisi faktor_inflasi & kurs_eur_idr (lihat ml/config.py).
    """
    F = fitur_dari_input(data)
    y_log = float(aktif.model.predict_log(F)[0])
    s = aktif.rmse_log_cv
    faktor = float(penyesuaian["faktor_inflasi"])
    kurs = float(penyesuaian["kurs_eur_idr"])

    eur_data = float(np.exp(y_log))
    bawah_data = float(np.exp(y_log - Z_95 * s))
    atas_data = float(np.exp(y_log + Z_95 * s))

    kontribusi = aktif.model.kontribusi(F)
    for item in kontribusi["fitur"]:
        item["label"] = LABEL_FITUR.get(item["fitur"], item["fitur"])

    pp = aktif.model.preprocessor
    return {
        "harga_dataset_eur": eur_data,
        "rentang_dataset_eur": [bawah_data, atas_data],
        "harga_sekarang_eur": eur_data * faktor,
        "rentang_sekarang_eur": [bawah_data * faktor, atas_data * faktor],
        "harga_idr": eur_data * faktor * kurs,
        "rentang_idr": [bawah_data * faktor * kurs, atas_data * faktor * kurs],
        "prediksi_log": y_log,
        "rentang": {
            "metode": "exp(ln_harga_prediksi ± 1,96 × RMSE_log hasil 5-Fold CV)",
            "z": Z_95,
            "rmse_log_cv": s,
        },
        "kontribusi": {
            **kontribusi,
            "acuan": {
                "kategori": dict(pp.kategori_acuan_),
                "keterangan": (
                    "Laptop acuan: semua fitur numerik bernilai rata-rata data latih, tanpa layar "
                    "sentuh & IPS, dan semua kategori = kategori acuan."
                ),
            },
        },
        "peringatan": _peringatan(aktif.model, data),
        "penyesuaian_harga": penyesuaian,
        "fitur_turunan": {"PPI": float(F["PPI"].iloc[0])},
    }
