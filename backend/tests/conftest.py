"""
conftest.py — Fixture bersama untuk pengujian.

- hasil_training : pipeline ML dijalankan SEKALI per sesi pengujian (±8 detik)
                   lalu hasilnya dipakai banyak tes.
- db_sementara   : database SQLite baru di folder sementara, tabelnya dibuat
                   lewat migrasi Alembic (sekaligus menguji file migrasi).
Database & folder model asli di backend/ tidak disentuh sama sekali.
"""

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.orm import sessionmaker

from db.database import buat_engine
from ml.config import FOLDER_BACKEND
from ml.train import jalankan_pipeline


@pytest.fixture(scope="session")
def hasil_training():
    hasil, model_terlatih = jalankan_pipeline(log=lambda *_: None)
    return hasil, model_terlatih


def jalankan_migrasi(url: str) -> None:
    konfigurasi = Config(str(FOLDER_BACKEND / "alembic.ini"))
    konfigurasi.set_main_option("sqlalchemy.url", url)
    command.upgrade(konfigurasi, "head")


@pytest.fixture
def db_sementara(tmp_path):
    url = f"sqlite:///{(tmp_path / 'uji.db').as_posix()}"
    jalankan_migrasi(url)
    engine = buat_engine(url)
    pembuat_session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)
    yield pembuat_session, tmp_path / "models"
    engine.dispose()
