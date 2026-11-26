"""
Pengujian penyimpanan hasil training ke database.

Yang dijamin:
- migrasi Alembic membuat keempat tabel,
- satu run training = satu nomor versi berisi 7 model,
- di seluruh tabel hanya ada SATU model aktif,
- semua JSON aman untuk PostgreSQL (tanpa NaN / tak hingga),
- file model yang disimpan menghasilkan prediksi yang sama persis.
"""

import json

import joblib
import numpy as np
from sqlalchemy import func, inspect, select

from db import repository
from db.tabel import EvaluationResult, ModelVersion, TrainingRun
from ml.model import SPESIFIKASI_MODEL


def test_migrasi_membuat_semua_tabel(db_sementara):
    pembuat_session, _ = db_sementara
    with pembuat_session() as s:
        tabel = set(inspect(s.get_bind()).get_table_names())
    assert {"training_runs", "model_versions", "evaluation_results", "predictions", "alembic_version"} <= tabel


def test_simpan_satu_run(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        run = repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        assert run.id == 1
        semua = s.scalars(select(ModelVersion)).all()
        assert {m.nama_model for m in semua} == set(SPESIFIKASI_MODEL)
        aktif = [m for m in semua if m.is_active]
        assert len(aktif) == 1 and aktif[0].nama_model == "ols_normal_equation"

        jenis_aktif = set(s.scalars(select(EvaluationResult.jenis).where(EvaluationResult.model_version_id == aktif[0].id)))
        assert {"metrik", "verifikasi", "cv", "residual", "qq_plot", "learning_curve", "error_per_merek",
                "error_per_tipe", "top_error", "koefisien"} <= jenis_aktif
        # kurva loss hanya dimiliki model Gradient Descent
        pemilik_loss = s.scalars(
            select(ModelVersion.nama_model).join(EvaluationResult).where(EvaluationResult.jenis == "loss_history")
        ).all()
        assert pemilik_loss == ["ols_gradient_descent"]


def test_run_kedua_menjadi_versi_baru_dan_hanya_satu_aktif(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        run2 = repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        assert run2.id == 2
        assert s.scalar(select(func.count()).where(ModelVersion.is_active.is_(True))) == 1
        assert repository.ambil_model_aktif(s).training_run_id == 2
        assert len(repository.daftar_model(s)) == 2 * len(SPESIFIKASI_MODEL)


def test_json_aman_untuk_postgresql(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        for data in s.scalars(select(EvaluationResult.data)):
            json.dumps(data, allow_nan=False)  # error jika ada NaN / Infinity
        for ringkasan in s.scalars(select(TrainingRun.ringkasan)):
            json.dumps(ringkasan, allow_nan=False)


def test_file_model_tersimpan_bisa_dimuat_ulang(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        aktif = repository.ambil_model_aktif(s)
        paket = joblib.load(repository.path_absolut(aktif.path_file))
    asli = model_terlatih["ols_normal_equation"]
    np.testing.assert_array_equal(paket["model"].regresor.coef_, asli.regresor.coef_)
    assert paket["rmse_log_cv"] > 0


def test_ambil_evaluasi_dan_prediksi(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        aktif = repository.ambil_model_aktif(s)
        evaluasi = repository.ambil_evaluasi(s, aktif.id)
        assert evaluasi["model"]["is_active"] is True
        assert len(evaluasi["perbandingan_run"]) == len(SPESIFIKASI_MODEL)
        assert evaluasi["ringkasan_run"]["pemilihan_model"]["model_terpilih"] == "ols_normal_equation"
        assert repository.ambil_evaluasi(s, 999_999) is None

        repository.simpan_prediksi(s, aktif.id, {"Ram": 8}, 900.0, 900.0 * 20010)
        riwayat = repository.riwayat_prediksi(s)
        assert riwayat["total"] == 1 and riwayat["data"][0]["input"] == {"Ram": 8}


def test_hapus_run_ikut_menghapus_model_dan_evaluasi(db_sementara, hasil_training):
    """Foreign key ON DELETE CASCADE juga berlaku di SQLite (PRAGMA foreign_keys)."""
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        run = repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
        s.delete(run)
        s.commit()
        assert s.scalar(select(func.count(ModelVersion.id))) == 0
        assert s.scalar(select(func.count(EvaluationResult.id))) == 0
