"""
tabel.py — Definisi tabel database (ORM SQLAlchemy).

Relasi antar tabel:

    training_runs (1) ──< model_versions (1) ──< evaluation_results
                                         (1) ──< predictions

- training_runs     : satu kali menjalankan `python -m ml.train` = satu
                      nomor versi. Menyimpan ringkasan yang berlaku untuk
                      seluruh run (info dataset, pemilihan alpha, verifikasi
                      koefisien, multikolinearitas, alasan pemilihan model).
                      (Tabel tambahan di luar 3 tabel minimal, agar data
                      tingkat-run tidak disalin ke setiap model.)
- model_versions    : satu model terlatih dalam satu run (7 model per run).
                      Hanya SATU baris di seluruh tabel yang is_active = True.
- evaluation_results: hasil evaluasi satu model, dipecah per jenis
                      (metrik, cv, residual, ...), isinya JSON.
- predictions       : riwayat prediksi dari website.

Kolom JSON memakai tipe JSON biasa di SQLite dan JSONB di PostgreSQL
(JSONB disimpan dalam bentuk biner sehingga lebih cepat dibaca).
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.database import Base

TipeJSON = JSON().with_variant(JSONB(), "postgresql")

# Daftar jenis hasil evaluasi yang sah (dicek di repository.py)
JENIS_EVALUASI = (
    "metrik",           # metrik data latih & uji, perbaikan vs baseline
    "verifikasi",       # metrik manual vs sklearn.metrics
    "cv",               # hasil 5-Fold Cross Validation per fold + ringkasan
    "residual",         # titik prediksi vs aktual, residual, histogram, statistik
    "qq_plot",          # titik Q-Q plot
    "learning_curve",   # error latih & validasi vs jumlah data
    "loss_history",     # kurva loss Gradient Descent (hanya model GD)
    "error_per_merek",
    "error_per_tipe",
    "top_error",        # 10 prediksi dengan error terbesar
    "koefisien",        # interpretasi koefisien
)


def _sekarang() -> datetime:
    return datetime.now(timezone.utc)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # = nomor versi
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_sekarang)
    durasi_detik: Mapped[float] = mapped_column(Float)
    ringkasan: Mapped[dict] = mapped_column(TipeJSON)

    model: Mapped[list["ModelVersion"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class ModelVersion(Base):
    __tablename__ = "model_versions"
    __table_args__ = (UniqueConstraint("training_run_id", "nama_model", name="uq_model_per_run"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    training_run_id: Mapped[int] = mapped_column(ForeignKey("training_runs.id", ondelete="CASCADE"), index=True)
    nama_model: Mapped[str] = mapped_column(String(64))
    label: Mapped[str] = mapped_column(String(128))
    hyperparameter: Mapped[dict] = mapped_column(TipeJSON, default=dict)
    path_file: Mapped[str] = mapped_column(String(512))  # relatif terhadap folder backend
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_sekarang)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    run: Mapped[TrainingRun] = relationship(back_populates="model")
    evaluasi: Mapped[list["EvaluationResult"]] = relationship(
        back_populates="model_version", cascade="all, delete-orphan"
    )
    prediksi: Mapped[list["Prediction"]] = relationship(back_populates="model_version")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"
    __table_args__ = (UniqueConstraint("model_version_id", "jenis", name="uq_jenis_per_model"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    jenis: Mapped[str] = mapped_column(String(32), index=True)
    data: Mapped[dict | list] = mapped_column(TipeJSON)

    model_version: Mapped[ModelVersion] = relationship(back_populates="evaluasi")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id", ondelete="CASCADE"), index=True)
    input: Mapped[dict] = mapped_column(TipeJSON)
    harga_prediksi_eur: Mapped[float] = mapped_column(Float)
    harga_prediksi_idr: Mapped[float] = mapped_column(Float)
    dibuat_pada: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_sekarang, index=True)

    model_version: Mapped[ModelVersion] = relationship(back_populates="prediksi")
