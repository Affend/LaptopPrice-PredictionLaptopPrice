"""
config.py — Satu tempat untuk semua pengaturan pipeline & aplikasi.

Nilai bisa diubah tanpa menyentuh kode, lewat variabel lingkungan atau file
`.env` (lihat .env.example di root proyek). Contoh:
    KURS_EUR_IDR=20500
    DATABASE_URL=postgresql+psycopg://user:pass@localhost:5432/laptopprice
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Lokasi folder penting (dihitung relatif terhadap file ini, jadi aman
# dijalankan dari folder mana pun)
FOLDER_BACKEND = Path(__file__).resolve().parent.parent
FOLDER_PROYEK = FOLDER_BACKEND.parent
FOLDER_DATA = FOLDER_BACKEND / "data"
FOLDER_MODEL = FOLDER_BACKEND / "models"
FOLDER_OUTPUT = FOLDER_BACKEND / "outputs"

# ---------------------------------------------------------------------------
# Konstanta eksperimen (sengaja TIDAK bisa diubah lewat .env agar hasil yang
# dilaporkan selalu bisa diulang persis sama)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42          # benih acak untuk split & K-Fold -> hasil bisa direproduksi
UKURAN_DATA_UJI = 0.2      # 20% data uji, 80% data latih
JUMLAH_FOLD = 5            # 5-Fold Cross Validation
Z_95 = 1.96                # nilai z untuk rentang kepercayaan ~95% (distribusi normal)


class Pengaturan(BaseSettings):
    """Pengaturan yang boleh diubah lewat variabel lingkungan / .env."""

    model_config = SettingsConfigDict(
        env_file=(FOLDER_PROYEK / ".env", FOLDER_BACKEND / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        # variabel yang ada tapi kosong (mis. KURS_EUR_IDR= dari docker-compose)
        # dianggap tidak diisi, sehingga nilai default yang dipakai
        env_ignore_empty=True,
    )

    # Database: default SQLite (file lokal) agar bisa jalan tanpa Docker
    database_url: str = f"sqlite:///{(FOLDER_BACKEND / 'laptopprice.db').as_posix()}"

    # Lokasi dataset
    path_dataset: Path = FOLDER_DATA / "laptop_price.csv"
    path_penyesuaian_harga: Path = FOLDER_DATA / "penyesuaian_harga.json"

    # Kurs & penyesuaian harga. Jika dikosongkan, nilai diambil dari
    # data/penyesuaian_harga.json (yang mencantumkan sumber dan tanggalnya).
    kurs_eur_idr: float | None = None
    faktor_inflasi: float | None = None

    # Origin frontend yang boleh memanggil API (CORS), dipisah koma
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"


@lru_cache
def ambil_pengaturan() -> Pengaturan:
    return Pengaturan()


def info_penyesuaian_harga(pengaturan: Pengaturan | None = None) -> dict:
    """Menghitung faktor inflasi & kurs yang dipakai, beserta sumbernya.

    faktor_inflasi = indeks HICP terkini / rata-rata indeks HICP tahun dasar

    Contoh: jika rata-rata indeks 2018 = 80 dan indeks terkini = 104, maka
    barang seharga 100 euro pada 2018 kira-kira setara 130 euro sekarang.
    """
    pengaturan = pengaturan or ambil_pengaturan()
    with open(pengaturan.path_penyesuaian_harga, encoding="utf-8") as f:
        data = json.load(f)

    bulanan = list(data["hicp"]["bulanan_tahun_dasar"].values())
    rata_rata_dasar = sum(bulanan) / len(bulanan)
    indeks_terkini = data["hicp"]["terkini"]["indeks"]
    faktor_dari_data = indeks_terkini / rata_rata_dasar

    return {
        "tahun_dasar_data": data["tahun_dasar_data"],
        "alasan_tahun_dasar": data["alasan_tahun_dasar"],
        "hicp_rata_rata_tahun_dasar": rata_rata_dasar,
        "hicp_terkini": indeks_terkini,
        "periode_hicp_terkini": data["hicp"]["terkini"]["periode"],
        "sumber_hicp": data["hicp"]["sumber"],
        "faktor_inflasi": pengaturan.faktor_inflasi or faktor_dari_data,
        "faktor_inflasi_diubah_manual": pengaturan.faktor_inflasi is not None,
        "kurs_eur_idr": pengaturan.kurs_eur_idr or data["kurs_eur_idr"]["nilai"],
        "kurs_diubah_manual": pengaturan.kurs_eur_idr is not None,
        "sumber_kurs": data["kurs_eur_idr"]["sumber"],
        "tanggal_kurs": data["kurs_eur_idr"]["tanggal"],
    }
