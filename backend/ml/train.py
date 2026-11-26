"""
train.py — Menjalankan seluruh pipeline ML: CSV -> model -> hasil evaluasi.

Cara menjalankan (dari folder backend, virtualenv aktif):
    alembic upgrade head          # sekali saja: buat tabel database
    python -m ml.train            # latih, simpan ke database sebagai versi baru
    python -m ml.train --no-db    # tanpa database: model ke models/lokal/

Hasil lengkap juga selalu ditulis ke outputs/hasil_evaluasi.json dan
gambar PNG ke outputs/ (bahan laporan).

Urutan kerja:
    1. Baca CSV, bersihkan, buat fitur, target = ln(harga).
    2. Split 80% latih / 20% uji (random_state = 42).
    3. Pilih alpha Ridge dengan 5-Fold CV (data latih saja).
    4. Untuk setiap model: 5-Fold CV, latih ulang di seluruh data latih,
       evaluasi di data uji, buat data grafik (residual, Q-Q, learning curve, ...).
    5. Verifikasi manual vs sklearn, analisis multikolinearitas & Gradient Descent.
    6. Pilih model terbaik = RMSE CV terkecil (BUKAN dari data uji).
    7. Simpan model (.joblib), hasil evaluasi, dan gambar PNG.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression

from ml import evaluation as ev
from ml.config import FOLDER_MODEL, FOLDER_OUTPUT, Z_95, ambil_pengaturan, info_penyesuaian_harga
from ml.model import SPESIFIKASI_MODEL, RegresiLinearGradientDescent, buat_model, tambah_kolom_bias
from ml.preprocessing import (
    FITUR_NUMERIK,
    FITUR_NUMERIK_SEMUA,
    FITUR_POLINOMIAL,
    Preprocessor,
    bersihkan_data,
    ekstrak_fitur,
    muat_dataset,
    target_log,
)

# ---------------------------------------------------------------------------
# Hyperparameter
# ---------------------------------------------------------------------------
# Gradient Descent: learning rate 0,1 berada jauh di bawah batas stabil
# 2/λ_max (≈ 0,33 untuk data ini, dihitung ulang & dilaporkan setiap training).
# Toleransi 1e-14: berhenti jika penurunan MSE antar iterasi < 1e-14.
HYPERPARAMETER_GD = {"learning_rate": 0.1, "n_iterasi": 20000, "toleransi": 1e-14}
# Kandidat alpha Ridge (kelipatan 10, dipilih lewat 5-Fold CV)
KANDIDAT_ALPHA = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
# Alpha untuk grafik "jalur koefisien Ridge" (efek penalti pada fitur kolinear)
ALPHA_JALUR_RIDGE = [0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]
# Learning rate yang dibandingkan di grafik kurva loss
LEARNING_RATE_PEMBANDING = [0.01, 0.05, 0.1, 0.3, 0.34]


def _turunkan_sampel(nilai: list[float], maks_titik: int = 400) -> dict:
    """Mengambil sebagian titik agar data grafik tidak terlalu besar."""
    n = len(nilai)
    if n <= maks_titik:
        idx = np.arange(n)
    else:
        # lebih rapat di awal (loss turun cepat), jarang di akhir
        idx = np.unique(np.concatenate([np.arange(50), np.geomspace(50, n - 1, maks_titik - 50).astype(int)]))
    return {"iterasi": [int(i) + 1 for i in idx], "loss": [float(nilai[i]) for i in idx]}


def _hyperparameter(nama: str, alpha: float) -> dict:
    if nama == "ols_gradient_descent":
        return dict(HYPERPARAMETER_GD)
    if nama in ("ridge_manual", "ridge_sklearn"):
        return {"alpha": alpha}
    if nama == "polinomial_d2":
        return {"derajat": 2, "fitur_polinomial": FITUR_POLINOMIAL}
    return {}


# ---------------------------------------------------------------------------
# Analisis tambahan
# ---------------------------------------------------------------------------
def analisis_gradient_descent(X: np.ndarray, y: np.ndarray, model_gd, model_ne) -> dict:
    """Data kurva loss + penjelasan konvergensi Gradient Descent.

    Hessian dari MSE adalah H = (2/n) XbᵀXb. GD stabil jika
    learning_rate < 2 / λ_max(H). Di atas batas itu loss membesar terus.
    """
    n = X.shape[0]
    Xb = tambah_kolom_bias(X)
    eigen = np.linalg.eigvalsh((2.0 / n) * Xb.T @ Xb)
    mse_ne = float(np.mean((model_ne.predict(X) - y) ** 2))

    perbandingan = []
    for lr in LEARNING_RATE_PEMBANDING:
        gd = RegresiLinearGradientDescent(learning_rate=lr, n_iterasi=3000, toleransi=None)
        try:
            gd.fit(X, y)
            riwayat = list(gd.riwayat_loss_)
        except ValueError:
            riwayat = []
        stabil = lr < 2 / eigen.max()
        # Untuk lr yang divergen, kurva dipotong saat loss melewati 10x loss
        # awal agar grafik tetap terbaca (setelah itu loss terus meledak).
        if not stabil and riwayat:
            batas = 10 * riwayat[0]
            potong = next((i for i, v in enumerate(riwayat) if v > batas), len(riwayat))
            riwayat = riwayat[: potong + 1]
        perbandingan.append(
            {
                "learning_rate": lr,
                "status": "konvergen" if stabil else "divergen",
                "loss_akhir": float(riwayat[-1]) if riwayat else None,
                **_turunkan_sampel(riwayat, 300),
            }
        )

    riwayat = model_gd.regresor.riwayat_loss_
    return {
        **_turunkan_sampel(riwayat),
        "learning_rate": HYPERPARAMETER_GD["learning_rate"],
        "n_iterasi_maks": HYPERPARAMETER_GD["n_iterasi"],
        "n_iterasi_dijalankan": int(model_gd.regresor.n_iterasi_dijalankan_),
        "loss_awal": float(riwayat[0]),
        "loss_akhir": float(riwayat[-1]),
        "loss_normal_equation": mse_ne,
        "eigen_hessian_maks": float(eigen.max()),
        "eigen_hessian_min": float(eigen.min()),
        "learning_rate_maks_stabil": float(2.0 / eigen.max()),
        "perbandingan_learning_rate": perbandingan,
    }


def analisis_multikolinearitas(df_mentah, F_latih, y_latih, alpha: float) -> dict:
    """Bukti angka untuk keputusan fitur & pembahasan multikolinearitas."""
    from ml.model import RegresiLinearNormalEquation, RegresiRidgeManual

    def kondisi(pp: Preprocessor, F) -> dict:
        X = pp.fit_transform(F)
        Xb = tambah_kolom_bias(X)
        rank = int(np.linalg.matrix_rank(Xb))
        return {
            "jumlah_kolom": int(Xb.shape[1]),
            "rank": rank,
            "singular": rank < Xb.shape[1],
            "condition_number": ev.condition_number(X),
        }

    F_mentah = ekstrak_fitur(df_mentah)
    skenario = [
        {
            "skenario": "Data mentah (laptop ARM ikut), kolom identik tidak dibuang, lebar & tinggi ikut",
            **kondisi(Preprocessor(fitur_numerik=FITUR_NUMERIK_SEMUA, buang_kolom_duplikat=False), F_mentah),
        },
        {
            "skenario": "Data bersih, kolom identik dibuang, lebar & tinggi ikut",
            **kondisi(Preprocessor(fitur_numerik=FITUR_NUMERIK_SEMUA), F_latih),
        },
        {
            "skenario": "Model akhir (lebar & tinggi diganti PPI)",
            **kondisi(Preprocessor(), F_latih),
        },
    ]
    X_akhir = Preprocessor().fit_transform(F_latih)
    for a in (alpha, 10.0, 100.0):
        skenario.append(
            {
                "skenario": f"Model akhir + Ridge (alpha = {a:g})" + (" — alpha terpilih CV" if a == alpha else ""),
                "jumlah_kolom": int(X_akhir.shape[1] + 1),
                "rank": int(X_akhir.shape[1] + 1),
                "singular": False,
                "condition_number": ev.condition_number(X_akhir, a),
            }
        )

    # Jalur koefisien Ridge untuk fitur layar ketika lebar, tinggi, PPI, dan
    # inci dipakai BERSAMAAN (VIF tinggi). Makin besar alpha, koefisien makin
    # kecil (menyusut ke nol) dan makin stabil. alpha = 0 sama dengan OLS.
    pp_semua = Preprocessor(fitur_numerik=FITUR_NUMERIK_SEMUA)
    X_semua = pp_semua.fit_transform(F_latih)
    fitur_layar = ["Inches", "Lebar_px", "Tinggi_px", "PPI"]
    jalur = []
    for a in ALPHA_JALUR_RIDGE:
        reg = RegresiLinearNormalEquation().fit(X_semua, y_latih) if a == 0 else RegresiRidgeManual(a).fit(X_semua, y_latih)
        jalur.append(
            {
                "alpha": a,
                **{f: float(reg.coef_[pp_semua.nama_fitur_.index(f)]) for f in fitur_layar},
                "norma_bobot": float(np.linalg.norm(reg.coef_)),
            }
        )

    pp_akhir = Preprocessor().fit(F_latih)
    return {
        "vif_semua_fitur_numerik": ev.hitung_vif(F_latih[FITUR_NUMERIK_SEMUA].to_numpy(float), FITUR_NUMERIK_SEMUA),
        "vif_fitur_dipakai": ev.hitung_vif(F_latih[FITUR_NUMERIK].to_numpy(float), FITUR_NUMERIK),
        "skenario_condition_number": skenario,
        "jalur_koefisien_ridge_fitur_layar": jalur,
        "kolom_identik_dibuang": pp_akhir.kolom_duplikat_,
        "korelasi_fitur_layar": F_latih[fitur_layar].corr().round(4).to_dict(),
    }


# ---------------------------------------------------------------------------
# Pemilihan model aktif
# ---------------------------------------------------------------------------
# Tingkat kompleksitas tiap model (makin kecil makin sederhana):
#   1 = regresi linear biasa (35 bobot, tanpa hyperparameter tambahan)
#   2 = Ridge (bobot sama banyak, tetapi ada hyperparameter alpha)
#   3 = polinomial (50 bobot: ditambah 15 suku kuadrat & interaksi)
# Jika kompleksitas sama, urutan di SPESIFIKASI_MODEL dipakai, sehingga
# implementasi manual (Normal Equation) didahulukan daripada sklearn.
KOMPLEKSITAS_MODEL = {
    "ols_normal_equation": 1,
    "ols_gradient_descent": 1,
    "ols_sklearn": 1,
    "ridge_manual": 2,
    "ridge_sklearn": 2,
    "polinomial_d2": 3,
}


def pilih_model_aturan_1se(hasil_model: dict) -> dict:
    """Aturan satu standard error (one-standard-error rule).

    1. Cari model dengan RMSE CV rata-rata terkecil.
    2. Hitung standard error RMSE model itu: SE = std_antar_fold / sqrt(k).
    3. Semua model dengan RMSE CV <= RMSE_terkecil + SE dianggap "setara
       secara statistik" (selisihnya masih dalam batas variasi antar fold).
    4. Dari model yang setara, pilih yang PALING SEDERHANA.

    Alasannya: model yang lebih sederhana lebih mudah ditafsirkan dan lebih
    kecil risikonya overfitting, sementara keunggulan kecil model kompleks
    di CV belum tentu nyata (bisa jadi hanya kebetulan pembagian fold).
    Sumber: Hastie, Tibshirani & Friedman, The Elements of Statistical
    Learning (2009), bagian 7.10.
    """
    urutan = list(SPESIFIKASI_MODEL)
    kandidat = [n for n in urutan if n in KOMPLEKSITAS_MODEL]

    def rmse_cv(n):
        return hasil_model[n]["cv"]["ringkasan"]["rmse"]["rata_rata"]

    terkecil = min(kandidat, key=rmse_cv)
    k = hasil_model[terkecil]["cv"]["k"]
    se = hasil_model[terkecil]["cv"]["ringkasan"]["rmse"]["std"] / np.sqrt(k)
    batas = rmse_cv(terkecil) + se
    setara = [n for n in kandidat if rmse_cv(n) <= batas]
    terpilih = min(setara, key=lambda n: (KOMPLEKSITAS_MODEL[n], urutan.index(n)))
    return {
        "aturan": "1-SE: model paling sederhana yang RMSE CV-nya <= RMSE CV terkecil + 1 standard error",
        "model_rmse_terkecil": terkecil,
        "rmse_terkecil": float(rmse_cv(terkecil)),
        "standard_error": float(se),
        "batas_1se": float(batas),
        "kandidat": [
            {
                "model": n,
                "rmse_cv": float(rmse_cv(n)),
                "kompleksitas": KOMPLEKSITAS_MODEL[n],
                "dalam_batas_1se": n in setara,
            }
            for n in kandidat
        ],
        "model_terpilih": terpilih,
    }


# ---------------------------------------------------------------------------
# Pipeline utama
# ---------------------------------------------------------------------------
def jalankan_pipeline(log=print) -> tuple[dict, dict]:
    """Menjalankan semua langkah. Mengembalikan (hasil_evaluasi, model_terlatih)."""
    t0 = time.time()
    pengaturan = ambil_pengaturan()

    # 1. Data
    df_mentah = muat_dataset(pengaturan.path_dataset)
    df, dibuang = bersihkan_data(df_mentah)
    F = ekstrak_fitur(df)
    y = target_log(df)
    log(f"Data: {len(df_mentah)} baris mentah, {len(dibuang)} dibuang, {len(df)} dipakai")

    # 2. Split latih / uji
    idx_latih, idx_uji = ev.bagi_latih_uji(len(df))
    F_latih, F_uji = F.iloc[idx_latih].reset_index(drop=True), F.iloc[idx_uji].reset_index(drop=True)
    y_latih, y_uji = y[idx_latih], y[idx_uji]
    info_uji = df.iloc[idx_uji].reset_index(drop=True)
    log(f"Split: {len(idx_latih)} latih, {len(idx_uji)} uji")

    # 3. Pilih alpha Ridge (CV di data latih)
    pemilihan_alpha = ev.pilih_alpha(
        lambda a: buat_model("ridge_manual", {"alpha": a}), F_latih, y_latih, KANDIDAT_ALPHA
    )
    alpha = pemilihan_alpha["alpha_terbaik"]
    log(f"Alpha Ridge terpilih (5-Fold CV): {alpha}")

    # 4. Setiap model
    hasil_model: dict[str, dict] = {}
    model_terlatih: dict[str, object] = {}
    for nama, spek in SPESIFIKASI_MODEL.items():
        hyper = _hyperparameter(nama, alpha)

        def pembuat(nama=nama, hyper=hyper):
            return buat_model(nama, hyper)

        cv = ev.validasi_silang(pembuat, F_latih, y_latih)
        model = pembuat().fit(F_latih, y_latih)
        pred_latih = model.predict_log(F_latih)
        pred_uji = model.predict_log(F_uji)

        hasil = {
            "label": spek["label"],
            "keterangan": spek["keterangan"],
            "hyperparameter": hyper,
            "jumlah_fitur": int(len(model.preprocessor.nama_fitur_)),
            "cv": cv,
            "metrik": {
                "latih": ev.evaluasi_prediksi(y_latih, pred_latih),
                "uji": ev.evaluasi_prediksi(y_uji, pred_uji),
            },
            "verifikasi_metrik": ev.verifikasi_metrik(np.exp(y_uji), np.exp(pred_uji)),
            "interval": ev.cakupan_interval(y_uji, pred_uji, cv["ringkasan"]["rmse_log"]["rata_rata"], Z_95),
            "residual": ev.data_residual(y_uji, pred_uji, info_uji),
            "qq_plot": ev.data_qq_plot(y_uji, pred_uji),
            "error_per_merek": ev.error_per_grup(info_uji, y_uji, pred_uji, "Company"),
            "error_per_tipe": ev.error_per_grup(info_uji, y_uji, pred_uji, "TypeName"),
            "top_error": ev.error_terbesar(info_uji, y_uji, pred_uji, n=10),
        }
        if nama != "baseline_rata_rata":
            hasil["koefisien"] = ev.interpretasi_koefisien(model)
            hasil["learning_curve"] = ev.kurva_belajar(pembuat, F_latih, y_latih)
        hasil_model[nama] = hasil
        model_terlatih[nama] = model
        r = cv["ringkasan"]
        log(
            f"  {nama:<22} CV RMSE {r['rmse']['rata_rata']:8.2f} ± {r['rmse']['std']:6.2f} | "
            f"CV MAPE {r['mape']['rata_rata']:6.2f}% | CV R² {r['r2']['rata_rata']:.4f} | "
            f"uji RMSE {hasil['metrik']['uji']['rmse']:8.2f}"
        )

    # Perbandingan dengan baseline (data uji & CV)
    base = hasil_model["baseline_rata_rata"]
    for hasil in hasil_model.values():
        hasil["perbaikan_vs_baseline"] = {
            m: {
                "uji_persen": (1 - hasil["metrik"]["uji"][m] / base["metrik"]["uji"][m]) * 100,
                "cv_persen": (1 - hasil["cv"]["ringkasan"][m]["rata_rata"] / base["cv"]["ringkasan"][m]["rata_rata"]) * 100,
            }
            for m in ("mae", "rmse", "mape")
        }

    # 5a. Verifikasi koefisien manual vs sklearn (preprocessor sama -> X sama)
    X_latih = model_terlatih["ols_normal_equation"].preprocessor.transform(F_latih)
    X_uji = model_terlatih["ols_normal_equation"].preprocessor.transform(F_uji)
    pp_poli = model_terlatih["polinomial_d2"].preprocessor
    poli_sklearn = LinearRegression().fit(pp_poli.transform(F_latih), y_latih)
    pasangan = [
        ("Normal Equation (manual) vs LinearRegression (sklearn)", "ols_normal_equation", "ols_sklearn", None),
        ("Gradient Descent (manual) vs Normal Equation (manual)", "ols_gradient_descent", "ols_normal_equation", None),
        ("Ridge manual vs Ridge sklearn", "ridge_manual", "ridge_sklearn", None),
        ("Polinomial manual vs LinearRegression sklearn (fitur polinomial sama)", "polinomial_d2", None, poli_sklearn),
    ]
    verifikasi_koefisien = []
    for judul, a, b, pembanding in pasangan:
        reg_b = pembanding if pembanding is not None else model_terlatih[b].regresor
        X_banding = pp_poli.transform(F_uji) if a == "polinomial_d2" else X_uji
        verifikasi_koefisien.append(
            {"perbandingan": judul, "model_a": a, "model_b": b or "sklearn_linear_polinomial",
             **ev.bandingkan_koefisien(model_terlatih[a].regresor, reg_b, X_banding)}
        )

    # 5b. Analisis Gradient Descent & multikolinearitas
    hasil_model["ols_gradient_descent"]["loss_history"] = analisis_gradient_descent(
        X_latih, y_latih, model_terlatih["ols_gradient_descent"], model_terlatih["ols_normal_equation"].regresor
    )
    multikolinearitas = analisis_multikolinearitas(df_mentah, F_latih, y_latih, alpha)

    # 6. Pilih model aktif dengan aturan 1-SE (one-standard-error rule)
    pemilihan = pilih_model_aturan_1se(hasil_model)
    terbaik = pemilihan["model_terpilih"]
    log(
        f"RMSE CV terkecil: {pemilihan['model_rmse_terkecil']} ({pemilihan['rmse_terkecil']:.2f}); "
        f"batas 1-SE: {pemilihan['batas_1se']:.2f}"
    )
    log(f"Model aktif (aturan 1-SE): {terbaik}")

    hasil = {
        "dibuat_pada": datetime.now(timezone.utc).isoformat(),
        "durasi_detik": round(time.time() - t0, 1),
        "dataset": {
            "jumlah_baris_mentah": int(len(df_mentah)),
            "baris_dibuang": dibuang,
            "jumlah_baris_dipakai": int(len(df)),
            "jumlah_latih": int(len(idx_latih)),
            "jumlah_uji": int(len(idx_uji)),
        },
        "pemilihan_alpha": pemilihan_alpha,
        "model": hasil_model,
        "verifikasi_koefisien": verifikasi_koefisien,
        "multikolinearitas": multikolinearitas,
        "model_terbaik": terbaik,
        "pemilihan_model": pemilihan,
        "penyesuaian_harga": info_penyesuaian_harga(pengaturan),
    }
    return hasil, model_terlatih


def simpan_model(model_terlatih: dict, hasil: dict, folder: Path) -> dict[str, str]:
    """Menyimpan setiap model terlatih ke file .joblib. Mengembalikan path-nya."""
    folder.mkdir(parents=True, exist_ok=True)
    path = {}
    for nama, model in model_terlatih.items():
        file = folder / f"{nama}.joblib"
        joblib.dump(
            {
                "model": model,
                "nama": nama,
                "label": hasil["model"][nama]["label"],
                # RMSE log dari CV dipakai API untuk rentang perkiraan harga
                "rmse_log_cv": hasil["model"][nama]["cv"]["ringkasan"]["rmse_log"]["rata_rata"],
                "dibuat_pada": hasil["dibuat_pada"],
            },
            file,
        )
        path[nama] = str(file)
    return path


def _ke_json(obj):
    """Mengubah tipe NumPy menjadi tipe Python biasa agar bisa ditulis ke JSON."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Tipe tidak dikenal: {type(obj)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Melatih & mengevaluasi model LaptopPrice")
    parser.add_argument("--no-db", action="store_true", help="jangan simpan ke database")
    parser.add_argument("--tanpa-gambar", action="store_true", help="lewati ekspor PNG")
    args = parser.parse_args(argv)

    hasil, model_terlatih = jalankan_pipeline()

    if args.no_db:
        hasil["path_model"] = simpan_model(model_terlatih, hasil, FOLDER_MODEL / "lokal")
    else:
        from db import repository
        from db.database import ambil_sessionmaker

        with ambil_sessionmaker()() as session:
            run = repository.simpan_hasil_training(session, hasil, model_terlatih, FOLDER_MODEL)
            aktif = repository.ambil_model_aktif(session)
            print(f"Tersimpan ke database sebagai versi {run.id}; model aktif: {aktif.nama_model} (id {aktif.id})")
            hasil["nomor_versi"] = run.id

    FOLDER_OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(FOLDER_OUTPUT / "hasil_evaluasi.json", "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=1, default=_ke_json)
    print(f"Hasil evaluasi: {FOLDER_OUTPUT / 'hasil_evaluasi.json'}")

    if not args.tanpa_gambar:
        from ml import plots

        df, _ = bersihkan_data(muat_dataset(ambil_pengaturan().path_dataset))
        file = plots.ekspor_semua(hasil, df, ekstrak_fitur(df), FOLDER_OUTPUT)
        print(f"{len(file)} gambar PNG disimpan di {FOLDER_OUTPUT}")
    print(f"Selesai dalam {hasil['durasi_detik']} detik")


if __name__ == "__main__":
    main()
