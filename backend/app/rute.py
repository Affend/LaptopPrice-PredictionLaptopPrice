"""
rute.py — Semua endpoint API.

Endpoint sengaja dibuat "tipis": hanya menerima request, memanggil fungsi
di layanan_*.py atau db/repository.py, lalu mengembalikan hasilnya.
Logika perhitungan tidak ditulis di sini.

    GET  /api/health               status API, database, dan model aktif
    GET  /api/dataset/summary      ringkasan & statistik dataset
    GET  /api/dataset/eda          data grafik eksplorasi
    GET  /api/dataset/laptop       cari laptop di dataset (isi form sekali klik)
    GET  /api/options              pilihan dropdown form prediksi
    POST /api/predict              prediksi harga + simpan ke riwayat
    GET  /api/models               daftar versi model & metrik
    GET  /api/evaluation/{id}      semua hasil evaluasi satu model
    GET  /api/predictions/history  riwayat prediksi
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.layanan_dataset import cari_laptop
from app.layanan_prediksi import hitung_prediksi
from app.skema import InputPrediksi, ResponsPrediksi
from db import repository

router = APIRouter(prefix="/api")


def ambil_session(request: Request):
    """Dependency FastAPI: satu sesi database per request, ditutup otomatis."""
    with request.app.state.pembuat_session() as session:
        yield session


def _model_aktif_atau_503(request: Request):
    aktif = request.app.state.model_aktif
    if aktif is None:
        raise HTTPException(status_code=503, detail=request.app.state.galat_model or "Model belum dimuat.")
    return aktif


@router.get("/health")
def health(request: Request, session: Session = Depends(ambil_session)):
    try:
        session.execute(text("SELECT 1"))
        status_db = "ok"
    except Exception as galat:  # noqa: BLE001 — sengaja menangkap semua galat koneksi
        status_db = f"galat: {galat.__class__.__name__}"
    aktif = request.app.state.model_aktif
    return {
        "status": "ok" if status_db == "ok" and aktif is not None else "bermasalah",
        "database": status_db,
        "model_aktif": aktif.info() if aktif else None,
        "pesan_model": request.app.state.galat_model,
        "jumlah_data": request.app.state.cache["summary"]["jumlah_baris_dipakai"],
    }


@router.get("/dataset/summary")
def dataset_summary(request: Request):
    return request.app.state.cache["summary"]


@router.get("/dataset/eda")
def dataset_eda(request: Request):
    return request.app.state.cache["eda"]


@router.get("/dataset/laptop")
def dataset_laptop(
    request: Request,
    q: str = Query("", max_length=100, description="kata kunci, mis. 'dell i7'"),
    batas: int = Query(12, ge=1, le=50),
):
    """Mencari laptop di dataset untuk mengisi form prediksi dengan sekali klik."""
    return cari_laptop(request.app.state.cache["laptop"], q, batas)


@router.get("/options")
def options(request: Request):
    aktif = request.app.state.model_aktif
    return {
        **request.app.state.cache["options"],
        "penyesuaian_harga": request.app.state.penyesuaian,
        # kategori yang benar-benar punya kolom sendiri di model aktif;
        # kategori lain digabung ke "Lainnya" / kategori acuan
        "kategori_model": aktif.model.preprocessor.kategori_ if aktif else None,
    }


@router.post("/predict", response_model=ResponsPrediksi)
def predict(input_: InputPrediksi, request: Request, session: Session = Depends(ambil_session)):
    aktif = _model_aktif_atau_503(request)

    # Nilai kategori harus salah satu pilihan yang ada di dataset
    opsi = request.app.state.cache["options"]
    kesalahan = []
    for field in ("company", "type_name", "cpu_kategori", "gpu_merek", "os"):
        sah = [o["nilai"] for o in opsi[field]]
        if getattr(input_, field) not in sah:
            kesalahan.append({"kolom": field, "pesan": f"Nilai tidak dikenal. Pilihan yang sah: {', '.join(sah)}"})
    if kesalahan:
        raise HTTPException(status_code=422, detail={"pesan": "Input tidak valid", "kesalahan": kesalahan})

    hasil = hitung_prediksi(aktif, input_.ke_kolom_model(), request.app.state.penyesuaian)
    tersimpan = repository.simpan_prediksi(
        session,
        model_version_id=aktif.id,
        input_={
            **input_.model_dump(),
            # dicatat agar angka rupiah di riwayat bisa ditelusuri ulang
            "faktor_inflasi": hasil["penyesuaian_harga"]["faktor_inflasi"],
            "kurs_eur_idr": hasil["penyesuaian_harga"]["kurs_eur_idr"],
        },
        eur=hasil["harga_dataset_eur"],
        idr=hasil["harga_idr"],
    )
    return {"prediksi_id": tersimpan.id, "model": aktif.info(), **hasil}


@router.get("/models")
def models(request: Request, session: Session = Depends(ambil_session)):
    aktif = request.app.state.model_aktif
    return {"model_aktif_id": aktif.id if aktif else None, "data": repository.daftar_model(session)}


@router.get("/evaluation/{model_id}")
def evaluation(model_id: int, session: Session = Depends(ambil_session)):
    hasil = repository.ambil_evaluasi(session, model_id)
    if hasil is None:
        raise HTTPException(status_code=404, detail=f"Model dengan id {model_id} tidak ditemukan.")
    return hasil


@router.get("/predictions/history")
def predictions_history(batas: int = Query(20, ge=1, le=100), session: Session = Depends(ambil_session)):
    return repository.riwayat_prediksi(session, batas)
