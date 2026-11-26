"""
siapkan.py — Dijalankan otomatis saat container backend dinyalakan (lihat Dockerfile).

Urutan:
    1. Tunggu sampai database (PostgreSQL) siap menerima koneksi.
    2. Jalankan migrasi Alembic -> tabel dibuat / diperbarui.
    3. Jika belum ada model aktif, ATAU file .joblib model aktif hilang
       (mis. volume model dihapus), jalankan training (python -m ml.train).
Setelah itu Dockerfile menyalakan server uvicorn.

Bisa juga dijalankan manual tanpa Docker:  python siapkan.py
"""

from __future__ import annotations

import sys
import time

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from db import repository
from db.database import ambil_engine, ambil_sessionmaker
from ml.config import FOLDER_BACKEND


def tunggu_database(batas_detik: int = 60) -> None:
    mulai = time.time()
    while True:
        try:
            with ambil_engine().connect() as koneksi:
                koneksi.execute(text("SELECT 1"))
            print("[siapkan] Database siap.")
            return
        except Exception as galat:  # noqa: BLE001
            if time.time() - mulai > batas_detik:
                print(f"[siapkan] Database tidak bisa dihubungi setelah {batas_detik} detik: {galat}")
                sys.exit(1)
            print("[siapkan] Menunggu database...")
            time.sleep(2)


def perlu_training() -> bool:
    with ambil_sessionmaker()() as session:
        aktif = repository.ambil_model_aktif(session)
        if aktif is None:
            print("[siapkan] Belum ada model aktif.")
            return True
        if not repository.path_absolut(aktif.path_file).exists():
            print(f"[siapkan] File model aktif hilang: {aktif.path_file}")
            return True
        print(f"[siapkan] Model aktif sudah ada: {aktif.nama_model} (versi {aktif.training_run_id}).")
        return False


def main() -> None:
    tunggu_database()
    print("[siapkan] Menjalankan migrasi Alembic...")
    command.upgrade(Config(str(FOLDER_BACKEND / "alembic.ini")), "head")
    if perlu_training():
        print("[siapkan] Menjalankan training (sekitar 10 detik)...")
        from ml.train import main as latih

        latih([])


if __name__ == "__main__":
    main()
