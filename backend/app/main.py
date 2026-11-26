"""
main.py — Titik masuk aplikasi FastAPI.

Menjalankan server (dari folder backend, virtualenv aktif):
    uvicorn app.main:app --reload --port 8000
Dokumentasi interaktif otomatis: http://localhost:8000/docs

Saat server dinyalakan (fungsi `siklus_hidup`), dilakukan SEKALI:
    1. baca dataset CSV, hitung ringkasan / EDA / pilihan form -> cache memori
    2. baca model aktif dari database, muat file .joblib-nya -> memori
Setiap request prediksi lalu memakai model yang sudah ada di memori, jadi
tidak perlu membaca file model berulang-ulang.

Catatan: setelah `python -m ml.train` menghasilkan model aktif baru,
server perlu dinyalakan ulang agar model baru dimuat.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, sessionmaker

from app import layanan_dataset
from app.layanan_prediksi import muat_model_aktif
from app.rute import router
from db.database import ambil_sessionmaker
from ml.config import ambil_pengaturan, info_penyesuaian_harga

log = logging.getLogger("laptopprice")


def buat_app(pembuat_session: sessionmaker[Session] | None = None, path_dataset: Path | None = None) -> FastAPI:
    """Membuat aplikasi. Parameter bisa diisi saat pengujian (database sementara)."""
    pengaturan = ambil_pengaturan()

    @asynccontextmanager
    async def siklus_hidup(app: FastAPI):
        state = app.state
        state.pembuat_session = pembuat_session or ambil_sessionmaker()

        # 1. Dataset -> cache
        data = layanan_dataset.muat_data_dataset(path_dataset or pengaturan.path_dataset)
        state.cache = {
            "summary": layanan_dataset.ringkasan(data),
            "eda": layanan_dataset.eda(data),
            "options": layanan_dataset.opsi_form(data),
            "laptop": layanan_dataset.daftar_laptop(data),
        }
        state.penyesuaian = info_penyesuaian_harga(pengaturan)

        # 2. Model aktif -> memori. Jika gagal, API tetap hidup (endpoint
        #    dataset & evaluasi masih bisa dipakai), /predict menjawab 503.
        state.model_aktif, state.galat_model = None, None
        try:
            with state.pembuat_session() as session:
                state.model_aktif = muat_model_aktif(session)
            if state.model_aktif is None:
                state.galat_model = "Belum ada model aktif di database. Jalankan: python -m ml.train"
        except Exception as galat:  # noqa: BLE001
            state.galat_model = (
                f"Gagal memuat model ({galat.__class__.__name__}). "
                "Pastikan database sudah dimigrasi (alembic upgrade head) lalu jalankan python -m ml.train."
            )
            log.exception("Gagal memuat model aktif")
        if state.model_aktif:
            log.info("Model aktif dimuat: %s (id %s)", state.model_aktif.nama_model, state.model_aktif.id)
        else:
            log.warning(state.galat_model)
        yield

    app = FastAPI(
        title="LaptopPrice API",
        description="Prediksi harga laptop berdasarkan spesifikasi menggunakan Regresi Linear Berganda.",
        version="1.0.0",
        lifespan=siklus_hidup,
    )

    # CORS: izinkan website Next.js (origin berbeda port) memanggil API ini
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[o.strip() for o in pengaturan.cors_origins.split(",") if o.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.exception_handler(RequestValidationError)
    async def _galat_validasi(_: Request, galat: RequestValidationError):
        """Pesan galat validasi yang lebih mudah dibaca frontend."""
        kesalahan = [
            {"kolom": ".".join(str(x) for x in e["loc"] if x != "body") or "input", "pesan": e["msg"]}
            for e in galat.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": {"pesan": "Input tidak valid", "kesalahan": kesalahan}})

    app.include_router(router)
    return app


app = buat_app()
