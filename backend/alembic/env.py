"""
env.py — Dijalankan Alembic setiap kali perintah migrasi dipanggil.

Alembic adalah "version control" untuk struktur database. Setiap perubahan
tabel ditulis sebagai file migrasi di alembic/versions/, sehingga database
di laptop anggota kelompok dan di Docker selalu punya struktur yang sama:
    alembic upgrade head      -> buat / perbarui tabel ke versi terbaru
    alembic downgrade -1      -> mundur satu migrasi
"""

from logging.config import fileConfig

from alembic import context

import db.tabel  # noqa: F401  (memastikan semua tabel terdaftar di Base.metadata)
from db.database import Base, buat_engine
from ml.config import ambil_pengaturan

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    # Bisa ditimpa dari kode (mis. pengujian): config.set_main_option("sqlalchemy.url", ...)
    return config.get_main_option("sqlalchemy.url") or ambil_pengaturan().database_url


def run_migrations_offline() -> None:
    """Mode offline: hanya mencetak perintah SQL tanpa terhubung ke database."""
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Mode online: terhubung ke database lalu menjalankan migrasi."""
    engine = buat_engine(_url())
    with engine.connect() as koneksi:
        # render_as_batch: SQLite tidak mendukung sebagian perintah ALTER TABLE;
        # mode batch membuat Alembic menyalin tabel agar migrasi tetap jalan.
        context.configure(connection=koneksi, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
