"""
database.py — Koneksi ke database lewat SQLAlchemy.

Alamat database diambil dari variabel DATABASE_URL (lihat ml/config.py):
    - Tanpa Docker  : sqlite:///.../backend/laptopprice.db  (default, cukup 1 file)
    - Dengan Docker : postgresql+psycopg://user:password@postgres:5432/laptopprice
Kode lain tidak perlu tahu database apa yang dipakai; SQLAlchemy yang
menerjemahkan perintah Python ke SQL yang sesuai.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from ml.config import ambil_pengaturan


class Base(DeclarativeBase):
    """Kelas induk semua tabel ORM."""


def buat_engine(url: str) -> Engine:
    """Membuat engine (pengelola koneksi) untuk alamat database tertentu."""
    if url.startswith("sqlite"):
        # check_same_thread=False: FastAPI bisa memakai koneksi dari thread lain
        engine = create_engine(url, connect_args={"check_same_thread": False})

        # SQLite secara bawaan TIDAK menegakkan foreign key. Perintah PRAGMA ini
        # menyalakannya agar perilakunya sama dengan PostgreSQL (mis. hapus
        # versi model -> hasil evaluasinya ikut terhapus).
        @event.listens_for(engine, "connect")
        def _nyalakan_foreign_key(koneksi_dbapi, _):
            kursor = koneksi_dbapi.cursor()
            kursor.execute("PRAGMA foreign_keys=ON")
            kursor.close()

        return engine
    # pool_pre_ping: cek koneksi masih hidup sebelum dipakai (mis. setelah
    # container PostgreSQL di-restart)
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def ambil_engine() -> Engine:
    """Engine untuk DATABASE_URL dari pengaturan (dibuat sekali, lalu dipakai ulang)."""
    return buat_engine(ambil_pengaturan().database_url)


def ambil_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=ambil_engine(), autoflush=False, expire_on_commit=False)
