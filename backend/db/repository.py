"""
repository.py — Semua operasi baca/tulis database dikumpulkan di sini.

train.py dan API FastAPI tidak menulis query sendiri; mereka memanggil
fungsi-fungsi di file ini. Dengan begitu aturan seperti "hanya satu model
yang aktif" dijaga di satu tempat saja.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from db.tabel import JENIS_EVALUASI, EvaluationResult, ModelVersion, Prediction, TrainingRun
from ml.config import FOLDER_BACKEND

# Bagian hasil evaluasi tiap model -> jenis di tabel evaluation_results
PEMETAAN_JENIS = {
    "verifikasi_metrik": "verifikasi",
    "cv": "cv",
    "residual": "residual",
    "qq_plot": "qq_plot",
    "learning_curve": "learning_curve",
    "loss_history": "loss_history",
    "error_per_merek": "error_per_merek",
    "error_per_tipe": "error_per_tipe",
    "top_error": "top_error",
    "koefisien": "koefisien",
}
# Bagian hasil training yang berlaku untuk seluruh run
KUNCI_RINGKASAN_RUN = [
    "dataset", "pemilihan_alpha", "verifikasi_koefisien", "multikolinearitas",
    "model_terbaik", "pemilihan_model", "penyesuaian_harga",
]


def bersihkan_json(obj):
    """Mengubah nilai agar aman disimpan sebagai JSON di SQLite & PostgreSQL.

    - Tipe NumPy (np.float64, np.int64, array) -> tipe Python biasa.
    - NaN / tak hingga -> None, karena PostgreSQL menolak NaN di kolom JSON.
    """
    if isinstance(obj, dict):
        return {str(k): bersihkan_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [bersihkan_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return bersihkan_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        nilai = float(obj)
        return nilai if math.isfinite(nilai) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _path_relatif(path: Path) -> str:
    """Path file model relatif terhadap folder backend (agar portabel ke Docker)."""
    path = Path(path).resolve()
    try:
        return path.relative_to(FOLDER_BACKEND).as_posix()
    except ValueError:
        return path.as_posix()


def path_absolut(path_file: str) -> Path:
    path = Path(path_file)
    return path if path.is_absolute() else FOLDER_BACKEND / path


# ---------------------------------------------------------------------------
# Menulis hasil training
# ---------------------------------------------------------------------------
def simpan_hasil_training(session: Session, hasil: dict, model_terlatih: dict, folder_model: Path) -> TrainingRun:
    """Menyimpan satu run training sebagai versi baru, lalu mengaktifkan model terpilih.

    Semua langkah berada dalam SATU transaksi: jika ada yang gagal di
    tengah jalan, tidak ada data setengah jadi yang tersimpan.
    """
    from ml.train import simpan_model  # import di sini untuk menghindari import melingkar

    run = TrainingRun(
        durasi_detik=hasil["durasi_detik"],
        ringkasan=bersihkan_json({k: hasil[k] for k in KUNCI_RINGKASAN_RUN}),
    )
    session.add(run)
    session.flush()  # flush -> database memberi nomor id (= nomor versi) tanpa commit

    path_file = simpan_model(model_terlatih, hasil, Path(folder_model) / f"v{run.id}")

    # Hanya boleh ada satu model aktif: matikan semua, lalu nyalakan yang terpilih
    session.execute(update(ModelVersion).values(is_active=False))
    for nama, hasil_model in hasil["model"].items():
        versi = ModelVersion(
            training_run_id=run.id,
            nama_model=nama,
            label=hasil_model["label"],
            hyperparameter=bersihkan_json(hasil_model["hyperparameter"]),
            path_file=_path_relatif(path_file[nama]),
            is_active=(nama == hasil["model_terbaik"]),
        )
        session.add(versi)
        session.flush()

        metrik = {
            "latih": hasil_model["metrik"]["latih"],
            "uji": hasil_model["metrik"]["uji"],
            "perbaikan_vs_baseline": hasil_model["perbaikan_vs_baseline"],
            "jumlah_fitur": hasil_model["jumlah_fitur"],
            "keterangan": hasil_model["keterangan"],
            "interval": hasil_model["interval"],
        }
        session.add(EvaluationResult(model_version_id=versi.id, jenis="metrik", data=bersihkan_json(metrik)))
        for kunci, jenis in PEMETAAN_JENIS.items():
            if kunci in hasil_model:
                session.add(
                    EvaluationResult(model_version_id=versi.id, jenis=jenis, data=bersihkan_json(hasil_model[kunci]))
                )
    session.commit()
    return run


# ---------------------------------------------------------------------------
# Membaca model & hasil evaluasi
# ---------------------------------------------------------------------------
def ambil_model_aktif(session: Session) -> ModelVersion | None:
    return session.scalars(select(ModelVersion).where(ModelVersion.is_active.is_(True))).first()


def _ambil_hasil(session: Session, model_ids: list[int], jenis: list[str]) -> dict[tuple[int, str], object]:
    baris = session.execute(
        select(EvaluationResult.model_version_id, EvaluationResult.jenis, EvaluationResult.data).where(
            EvaluationResult.model_version_id.in_(model_ids), EvaluationResult.jenis.in_(jenis)
        )
    ).all()
    return {(mid, j): data for mid, j, data in baris}


def _ringkas_model(versi: ModelVersion, metrik: dict | None, cv: dict | None) -> dict:
    return {
        "id": versi.id,
        "nomor_versi": versi.training_run_id,
        "nama_model": versi.nama_model,
        "label": versi.label,
        "hyperparameter": versi.hyperparameter,
        "dibuat_pada": versi.dibuat_pada.isoformat() if versi.dibuat_pada else None,
        "is_active": versi.is_active,
        "metrik_uji": (metrik or {}).get("uji"),
        "metrik_latih": (metrik or {}).get("latih"),
        "perbaikan_vs_baseline": (metrik or {}).get("perbaikan_vs_baseline"),
        "jumlah_fitur": (metrik or {}).get("jumlah_fitur"),
        "keterangan": (metrik or {}).get("keterangan"),
        "interval": (metrik or {}).get("interval"),
        "cv": (cv or {}).get("ringkasan"),
    }


def daftar_model(session: Session) -> list[dict]:
    """Semua versi model beserta metrik ringkasnya (terbaru di atas)."""
    semua = session.scalars(
        select(ModelVersion).order_by(ModelVersion.training_run_id.desc(), ModelVersion.id)
    ).all()
    hasil = _ambil_hasil(session, [m.id for m in semua], ["metrik", "cv"])
    return [_ringkas_model(m, hasil.get((m.id, "metrik")), hasil.get((m.id, "cv"))) for m in semua]


def ambil_evaluasi(session: Session, model_id: int) -> dict | None:
    """Semua hasil evaluasi satu model + perbandingan dengan model lain di run yang sama."""
    versi = session.get(ModelVersion, model_id)
    if versi is None:
        return None
    evaluasi = {
        hasil.jenis: hasil.data
        for hasil in session.scalars(select(EvaluationResult).where(EvaluationResult.model_version_id == model_id))
    }
    satu_run = session.scalars(
        select(ModelVersion).where(ModelVersion.training_run_id == versi.training_run_id).order_by(ModelVersion.id)
    ).all()
    hasil_run = _ambil_hasil(session, [m.id for m in satu_run], ["metrik", "cv"])
    return {
        "model": _ringkas_model(versi, evaluasi.get("metrik"), evaluasi.get("cv")),
        "evaluasi": evaluasi,
        "perbandingan_run": [
            _ringkas_model(m, hasil_run.get((m.id, "metrik")), hasil_run.get((m.id, "cv"))) for m in satu_run
        ],
        "ringkasan_run": {
            "nomor_versi": versi.run.id,
            "dibuat_pada": versi.run.dibuat_pada.isoformat() if versi.run.dibuat_pada else None,
            "durasi_detik": versi.run.durasi_detik,
            **versi.run.ringkasan,
        },
    }


# ---------------------------------------------------------------------------
# Prediksi
# ---------------------------------------------------------------------------
def simpan_prediksi(session: Session, model_version_id: int, input_: dict, eur: float, idr: float) -> Prediction:
    prediksi = Prediction(
        model_version_id=model_version_id,
        input=bersihkan_json(input_),
        harga_prediksi_eur=float(eur),
        harga_prediksi_idr=float(idr),
    )
    session.add(prediksi)
    session.commit()
    return prediksi


def riwayat_prediksi(session: Session, batas: int = 20) -> dict:
    total = session.scalar(select(func.count(Prediction.id)))
    baris = session.execute(
        select(Prediction, ModelVersion.label)
        .join(ModelVersion, Prediction.model_version_id == ModelVersion.id)
        .order_by(Prediction.dibuat_pada.desc(), Prediction.id.desc())
        .limit(batas)
    ).all()
    return {
        "total": int(total or 0),
        "data": [
            {
                "id": p.id,
                "model_version_id": p.model_version_id,
                "label_model": label,
                "input": p.input,
                "harga_prediksi_eur": p.harga_prediksi_eur,
                "harga_prediksi_idr": p.harga_prediksi_idr,
                "dibuat_pada": p.dibuat_pada.isoformat() if p.dibuat_pada else None,
            }
            for p, label in baris
        ],
    }


def jenis_valid(jenis: str) -> bool:
    return jenis in JENIS_EVALUASI
