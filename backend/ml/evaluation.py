"""
evaluation.py — Validasi & evaluasi model (bagian terpenting tugas).

Isi file:
    Bagian 1: metrik MAE, RMSE, MAPE, R² dihitung MANUAL dengan NumPy, lalu
              dicek ulang dengan sklearn.metrics agar terbukti rumusnya benar.
    Bagian 2: skema validasi (split latih/uji, K-Fold CV, pemilihan alpha,
              learning curve).
    Bagian 3: diagnostik residual (data grafik residual, histogram, Q-Q plot).
    Bagian 4: analisis error (per merek/tipe, 10 error terbesar) dan
              interpretasi koefisien.
    Bagian 5: verifikasi manual vs sklearn dan analisis multikolinearitas.

Ringkasan alur validasi (dijalankan oleh train.py):
    1. Data dibagi 80% latih / 20% uji (random_state tetap -> bisa diulang).
    2. Semua keputusan (memilih alpha Ridge, memilih model terbaik) diambil
       HANYA dari 5-Fold Cross Validation di data latih.
    3. Data uji baru dipakai di akhir, satu kali, untuk melaporkan performa
       pada data yang benar-benar belum pernah dilihat model.
    Jika data uji ikut dipakai untuk memilih model, skor ujinya menjadi
    terlalu optimis karena model "dipilih supaya cocok" dengan data uji.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics as skm
from sklearn.model_selection import KFold, train_test_split

from ml.config import JUMLAH_FOLD, RANDOM_STATE, UKURAN_DATA_UJI


# ===========================================================================
# BAGIAN 1 — METRIK MANUAL
# ===========================================================================
# Notasi: y = nilai aktual, y_hat = nilai prediksi, n = jumlah data.
# Semua metrik di bawah dihitung pada skala HARGA ASLI (euro), yaitu setelah
# prediksi ln(harga) dikembalikan dengan exp(). Dengan begitu angkanya bisa
# dibaca langsung: "rata-rata meleset sekian euro".


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean Absolute Error = (1/n) * sum |y - y_hat|

    Rata-rata besar kesalahan, dalam satuan yang sama dengan harga (euro).
    Tidak terlalu terpengaruh oleh beberapa error yang sangat besar.
    """
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean(np.abs(y - y_hat)))


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean Squared Error = (1/n) * sum (y - y_hat)^2

    Fungsi loss yang diminimalkan oleh regresi linear (pada skala log).
    """
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean((y - y_hat) ** 2))


def rmse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Root Mean Squared Error = sqrt(MSE)

    Satuannya euro seperti MAE, tetapi error besar diberi bobot lebih berat
    karena dikuadratkan dulu. RMSE >= MAE selalu; jika RMSE jauh lebih besar
    dari MAE berarti ada beberapa prediksi yang meleset sangat jauh.
    """
    return float(np.sqrt(mse(y, y_hat)))


def mape(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean Absolute Percentage Error = (100/n) * sum |(y - y_hat) / y|

    Rata-rata kesalahan dalam persen terhadap harga asli. Cocok untuk harga
    karena salah 100 euro pada laptop 300 euro jauh lebih parah daripada
    salah 100 euro pada laptop 3000 euro. Aman dipakai di sini karena tidak
    ada harga bernilai 0.

    Catatan: sklearn mengembalikan MAPE dalam bentuk pecahan (0.15), di sini
    dikali 100 agar terbaca sebagai persen (15%).
    """
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    return float(np.mean(np.abs((y - y_hat) / y)) * 100.0)


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Koefisien determinasi R^2 = 1 - SSE / SST

    SSE = sum (y - y_hat)^2      -> error model
    SST = sum (y - rata2(y))^2   -> error jika selalu menebak rata-rata

    R^2 = 1   -> prediksi sempurna
    R^2 = 0   -> sama saja dengan menebak rata-rata (baseline)
    R^2 < 0   -> lebih buruk dari menebak rata-rata
    Artinya: proporsi variasi harga yang berhasil dijelaskan oleh model.
    """
    y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
    sse = np.sum((y - y_hat) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    return float(1.0 - sse / sst)


def hitung_semua_metrik(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Menghitung keempat metrik utama sekaligus (versi manual)."""
    return {
        "mae": mae(y, y_hat),
        "rmse": rmse(y, y_hat),
        "mape": mape(y, y_hat),
        "r2": r2(y, y_hat),
    }


def hitung_metrik_sklearn(y: np.ndarray, y_hat: np.ndarray) -> dict[str, float]:
    """Metrik yang sama, tetapi dihitung dengan sklearn.metrics (pembanding)."""
    return {
        "mae": float(skm.mean_absolute_error(y, y_hat)),
        "rmse": float(np.sqrt(skm.mean_squared_error(y, y_hat))),
        "mape": float(skm.mean_absolute_percentage_error(y, y_hat) * 100.0),
        "r2": float(skm.r2_score(y, y_hat)),
    }


def verifikasi_metrik(y: np.ndarray, y_hat: np.ndarray) -> list[dict]:
    """Membandingkan metrik manual dengan sklearn, baris per baris.

    Mengembalikan daftar seperti:
        [{"metrik": "mae", "manual": 150.2, "sklearn": 150.2,
          "selisih_absolut": 0.0, "sama": True}, ...]
    "sama" bernilai True jika selisihnya hanya sebatas pembulatan
    floating point (np.isclose).
    """
    manual = hitung_semua_metrik(y, y_hat)
    pembanding = hitung_metrik_sklearn(y, y_hat)
    hasil = []
    for nama in manual:
        selisih = abs(manual[nama] - pembanding[nama])
        hasil.append(
            {
                "metrik": nama,
                "manual": manual[nama],
                "sklearn": pembanding[nama],
                "selisih_absolut": selisih,
                "sama": bool(np.isclose(manual[nama], pembanding[nama], rtol=1e-9, atol=1e-9)),
            }
        )
    return hasil


# ===========================================================================
# BAGIAN 2 — SKEMA VALIDASI
# ===========================================================================
def bagi_latih_uji(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Mengembalikan indeks baris data latih (80%) dan data uji (20%)."""
    semua = np.arange(n)
    idx_latih, idx_uji = train_test_split(semua, test_size=UKURAN_DATA_UJI, random_state=RANDOM_STATE)
    return idx_latih, idx_uji


def evaluasi_prediksi(y_log_aktual: np.ndarray, y_log_prediksi: np.ndarray) -> dict[str, float]:
    """Metrik pada skala harga asli (euro) + R² & RMSE pada skala log.

    Model dilatih pada ln(harga), jadi prediksi dikembalikan dulu dengan
    exp() sebelum dibandingkan dengan harga aktual. R² log dicatat karena
    itulah besaran yang benar-benar dioptimalkan oleh regresi, dan RMSE log
    dipakai untuk menghitung rentang perkiraan harga di API.
    """
    aktual = np.exp(y_log_aktual)
    prediksi = np.exp(y_log_prediksi)
    hasil = hitung_semua_metrik(aktual, prediksi)
    hasil["r2_log"] = r2(y_log_aktual, y_log_prediksi)
    hasil["rmse_log"] = rmse(y_log_aktual, y_log_prediksi)
    return hasil


def cakupan_interval(y_log_aktual: np.ndarray, y_log_prediksi: np.ndarray, rmse_log: float, z: float) -> dict:
    """Menguji rentang perkiraan harga yang ditampilkan API.

    API menampilkan rentang  exp(y_hat - z*RMSE_log) .. exp(y_hat + z*RMSE_log),
    dengan RMSE_log dari Cross Validation. Jika residual log kira-kira normal,
    sekitar 95% harga aktual (z = 1,96) seharusnya jatuh di dalam rentang.
    Fungsi ini menghitung persentase SEBENARNYA di data uji, sehingga klaim
    "rentang 95%" dibuktikan dengan data, bukan hanya diasumsikan.
    """
    bawah = y_log_prediksi - z * rmse_log
    atas = y_log_prediksi + z * rmse_log
    di_dalam = (y_log_aktual >= bawah) & (y_log_aktual <= atas)
    return {
        "z": z,
        "rmse_log_cv": float(rmse_log),
        "target_cakupan_persen": float((2 * stats.norm.cdf(z) - 1) * 100),
        "cakupan_uji_persen": float(di_dalam.mean() * 100),
        "jumlah_di_luar": int((~di_dalam).sum()),
        "jumlah_uji": int(len(di_dalam)),
        # rentang relatif terhadap harga prediksi, mis. -40% .. +66%
        "batas_bawah_relatif_persen": float((np.exp(-z * rmse_log) - 1) * 100),
        "batas_atas_relatif_persen": float((np.exp(z * rmse_log) - 1) * 100),
    }


def _ringkas_fold(per_fold: list[dict]) -> dict:
    """Rata-rata dan standar deviasi (sampel, ddof=1) tiap metrik antar fold."""
    nama_metrik = [k for k in per_fold[0] if k != "fold"]
    return {
        m: {
            "rata_rata": float(np.mean([f[m] for f in per_fold])),
            "std": float(np.std([f[m] for f in per_fold], ddof=1)),
        }
        for m in nama_metrik
    }


def validasi_silang(
    pembuat_model: Callable[[], object],
    F_latih: pd.DataFrame,
    y_latih: np.ndarray,
    k: int = JUMLAH_FOLD,
) -> dict:
    """K-Fold Cross Validation.

    Data latih dibagi menjadi k bagian. Bergantian, 1 bagian menjadi data
    validasi dan k-1 bagian lain menjadi data latih. Hasilnya k skor; rata-
    ratanya menunjukkan performa yang diharapkan, dan standar deviasinya
    menunjukkan seberapa stabil model terhadap perubahan data.

    `pembuat_model` adalah fungsi yang menghasilkan model BARU, sehingga
    setiap fold (termasuk preprocessing-nya) dilatih dari nol.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    per_fold = []
    for nomor, (idx_a, idx_b) in enumerate(kf.split(F_latih), start=1):
        model = pembuat_model().fit(F_latih.iloc[idx_a], y_latih[idx_a])
        skor_validasi = evaluasi_prediksi(y_latih[idx_b], model.predict_log(F_latih.iloc[idx_b]))
        skor_latih = evaluasi_prediksi(y_latih[idx_a], model.predict_log(F_latih.iloc[idx_a]))
        per_fold.append(
            {
                "fold": nomor,
                **skor_validasi,
                "rmse_latih": skor_latih["rmse"],
                "r2_latih": skor_latih["r2"],
            }
        )
    return {"k": k, "per_fold": per_fold, "ringkasan": _ringkas_fold(per_fold)}


def pilih_alpha(
    pembuat_dengan_alpha: Callable[[float], object],
    F_latih: pd.DataFrame,
    y_latih: np.ndarray,
    kandidat: list[float],
) -> dict:
    """Mencoba beberapa nilai alpha Ridge, pilih yang RMSE CV-nya terkecil."""
    hasil = []
    for alpha in kandidat:
        cv = validasi_silang(lambda: pembuat_dengan_alpha(alpha), F_latih, y_latih)
        hasil.append(
            {
                "alpha": alpha,
                "rmse_cv": cv["ringkasan"]["rmse"]["rata_rata"],
                "rmse_cv_std": cv["ringkasan"]["rmse"]["std"],
                "mape_cv": cv["ringkasan"]["mape"]["rata_rata"],
                "r2_cv": cv["ringkasan"]["r2"]["rata_rata"],
            }
        )
    terbaik = min(hasil, key=lambda h: h["rmse_cv"])
    return {"kandidat": hasil, "alpha_terbaik": terbaik["alpha"]}


def kurva_belajar(
    pembuat_model: Callable[[], object],
    F_latih: pd.DataFrame,
    y_latih: np.ndarray,
    proporsi: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
) -> list[dict]:
    """Learning curve: bagaimana error berubah jika data latih ditambah.

    Untuk tiap proporsi p, di setiap fold CV model dilatih dengan p bagian
    dari data latih fold tersebut, lalu dihitung RMSE pada data yang dipakai
    melatih (error latih) dan pada data validasi fold (error validasi).

    Cara membaca:
    - Kedua kurva tinggi & berdekatan -> underfitting (model terlalu sederhana).
    - Kurva latih rendah, validasi jauh lebih tinggi -> overfitting.
    - Kurva validasi masih turun di ujung kanan -> menambah data masih membantu.
    """
    kf = KFold(n_splits=JUMLAH_FOLD, shuffle=True, random_state=RANDOM_STATE)
    rng = np.random.default_rng(RANDOM_STATE)
    lipatan = [(rng.permutation(a), b) for a, b in kf.split(F_latih)]
    hasil = []
    for p in proporsi:
        rmse_latih, rmse_validasi = [], []
        for idx_a, idx_b in lipatan:
            sub = idx_a[: max(int(round(p * len(idx_a))), 30)]
            model = pembuat_model().fit(F_latih.iloc[sub], y_latih[sub])
            rmse_latih.append(evaluasi_prediksi(y_latih[sub], model.predict_log(F_latih.iloc[sub]))["rmse"])
            rmse_validasi.append(
                evaluasi_prediksi(y_latih[idx_b], model.predict_log(F_latih.iloc[idx_b]))["rmse"]
            )
        hasil.append(
            {
                "proporsi": p,
                "jumlah_data_latih": int(len(sub)),
                "rmse_latih": float(np.mean(rmse_latih)),
                "rmse_latih_std": float(np.std(rmse_latih, ddof=1)),
                "rmse_validasi": float(np.mean(rmse_validasi)),
                "rmse_validasi_std": float(np.std(rmse_validasi, ddof=1)),
            }
        )
    return hasil


# ===========================================================================
# BAGIAN 3 — DIAGNOSTIK RESIDUAL (data untuk grafik)
# ===========================================================================
def data_residual(y_log_aktual: np.ndarray, y_log_prediksi: np.ndarray, info_baris: pd.DataFrame) -> dict:
    """Data grafik prediksi vs aktual, residual vs prediksi, dan histogram residual.

    Residual = aktual - prediksi.
    - residual_log: pada skala ln(harga). Inilah residual yang diasumsikan
      normal & variansinya konstan oleh regresi linear, jadi diagnostik
      (histogram, Q-Q plot, uji normalitas) memakai residual ini.
    - residual_euro: selisih dalam euro, supaya mudah dibaca.
    Residual positif = model menebak terlalu murah; negatif = terlalu mahal.
    """
    aktual = np.exp(y_log_aktual)
    prediksi = np.exp(y_log_prediksi)
    res_log = y_log_aktual - y_log_prediksi

    titik = [
        {
            "aktual_euro": float(a),
            "prediksi_euro": float(p),
            "residual_euro": float(a - p),
            "residual_log": float(r),
            "company": str(c),
            "product": str(prod),
            "typename": str(t),
        }
        for a, p, r, c, prod, t in zip(
            aktual, prediksi, res_log, info_baris["Company"], info_baris["Product"], info_baris["TypeName"]
        )
    ]

    # Histogram residual log + kurva normal pembanding dengan mean & std yang sama
    frekuensi, tepi = np.histogram(res_log, bins=20)
    lebar_bin = tepi[1] - tepi[0]
    mu, sigma = float(np.mean(res_log)), float(np.std(res_log, ddof=1))
    histogram = []
    for i in range(len(frekuensi)):
        tengah = (tepi[i] + tepi[i + 1]) / 2
        histogram.append(
            {
                "bin_kiri": float(tepi[i]),
                "bin_kanan": float(tepi[i + 1]),
                "tengah": float(tengah),
                "frekuensi": int(frekuensi[i]),
                # frekuensi yang diharapkan JIKA residual berdistribusi normal
                "frekuensi_normal": float(len(res_log) * lebar_bin * stats.norm.pdf(tengah, mu, sigma)),
            }
        )

    uji_normal = stats.shapiro(res_log)
    statistik = {
        "n": int(len(res_log)),
        "rata_rata_residual_log": mu,
        "std_residual_log": sigma,
        "skewness": float(stats.skew(res_log)),
        "kurtosis_excess": float(stats.kurtosis(res_log)),
        # Uji Shapiro-Wilk: H0 = residual berdistribusi normal.
        # p < 0,05 -> H0 ditolak (residual tidak normal sempurna).
        "shapiro_w": float(uji_normal.statistic),
        "shapiro_p": float(uji_normal.pvalue),
        # Korelasi |residual| dengan prediksi: mendekati 0 berarti besar
        # error tidak bergantung pada besar harga (homoskedastis).
        "korelasi_abs_residual_vs_prediksi": float(np.corrcoef(np.abs(res_log), y_log_prediksi)[0, 1]),
    }
    return {"titik": titik, "histogram": histogram, "statistik": statistik}


def data_qq_plot(y_log_aktual: np.ndarray, y_log_prediksi: np.ndarray) -> dict:
    """Q-Q plot residual terhadap distribusi normal.

    Residual distandarkan lalu diurutkan (kuantil sampel), kemudian
    dipasangkan dengan kuantil teoretis normal baku pada posisi
    (i - 0,5) / n. Jika residual normal, titik-titik mengikuti garis y = x.
    Ujung yang melengkung menjauhi garis = ekor distribusi lebih tebal
    (lebih banyak error ekstrem daripada yang diharapkan distribusi normal).
    """
    res = y_log_aktual - y_log_prediksi
    z = np.sort((res - res.mean()) / res.std(ddof=1))
    n = len(z)
    teoretis = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    return {"titik": [{"teoretis": float(t), "sampel": float(s)} for t, s in zip(teoretis, z)]}


# ===========================================================================
# BAGIAN 4 — ANALISIS ERROR & INTERPRETASI KOEFISIEN
# ===========================================================================
def error_per_grup(info_baris: pd.DataFrame, y_log_aktual, y_log_prediksi, kolom: str) -> list[dict]:
    """MAE, MAPE, dan bias per kelompok (mis. per merek atau per tipe laptop).

    bias_rata_rata = rata-rata (prediksi - aktual). Positif = model cenderung
    menebak terlalu mahal untuk kelompok itu; negatif = terlalu murah.
    """
    tabel = pd.DataFrame(
        {
            "grup": info_baris[kolom].astype(str).to_numpy(),
            "aktual": np.exp(y_log_aktual),
            "prediksi": np.exp(y_log_prediksi),
        }
    )
    tabel["error_abs"] = (tabel["aktual"] - tabel["prediksi"]).abs()
    tabel["error_persen"] = tabel["error_abs"] / tabel["aktual"] * 100
    tabel["bias"] = tabel["prediksi"] - tabel["aktual"]
    ringkas = (
        tabel.groupby("grup")
        .agg(
            n=("aktual", "size"),
            rata_rata_aktual=("aktual", "mean"),
            rata_rata_prediksi=("prediksi", "mean"),
            mae=("error_abs", "mean"),
            mape=("error_persen", "mean"),
            bias_rata_rata=("bias", "mean"),
        )
        .reset_index()
        .sort_values(["n", "grup"], ascending=[False, True])
    )
    return [
        {k: (v if isinstance(v, str) else float(v)) for k, v in baris.items()}
        for baris in ringkas.to_dict("records")
    ]


def error_terbesar(info_baris: pd.DataFrame, y_log_aktual, y_log_prediksi, n: int = 10) -> list[dict]:
    """Daftar n prediksi dengan error absolut (euro) terbesar."""
    aktual = np.exp(y_log_aktual)
    prediksi = np.exp(y_log_prediksi)
    error = np.abs(aktual - prediksi)
    urutan = np.argsort(-error)[:n]
    kolom_info = ["Company", "Product", "TypeName", "CPU_Type", "Ram", "Memory", "GPU_Type", "OpSys"]
    hasil = []
    for peringkat, i in enumerate(urutan, start=1):
        baris = {k.lower(): str(info_baris.iloc[i][k]) for k in kolom_info if k in info_baris.columns}
        baris.update(
            {
                "peringkat": peringkat,
                "aktual_euro": float(aktual[i]),
                "prediksi_euro": float(prediksi[i]),
                "error_euro": float(error[i]),
                "error_persen": float(error[i] / aktual[i] * 100),
                "arah": "terlalu murah" if prediksi[i] < aktual[i] else "terlalu mahal",
            }
        )
        hasil.append(baris)
    return hasil


def interpretasi_koefisien(model) -> list[dict]:
    """Menerjemahkan koefisien model ke "efek terhadap harga" (dalam persen).

    Karena target = ln(harga), koefisien w berarti: jika fitur naik 1 satuan
    (fitur lain tetap), harga dikali exp(w), yaitu berubah (exp(w) - 1) x 100%.
    - Fitur numerik: 1 satuan = 1 standar deviasi data latih (karena
      distandardisasi). Nilai std dalam satuan fitur ikut ditampilkan.
    - Fitur log2 (RAM, SSD, HDD, Flash): ditambah efek "per penggandaan".
      Penggandaan = log2 naik 1 = fitur terstandar naik 1/std, sehingga
      efeknya exp(w / std) - 1.
    - Fitur biner (Touchscreen, IPS): efek jika fitur ada dibanding tidak.
    - Kolom one-hot: efek kategori dibanding kategori acuan.
    """
    pp = model.preprocessor
    std_asli = dict(zip(pp.fitur_numerik, pp.std_))
    hasil = []
    for nama, w in zip(pp.nama_fitur_, model.regresor.coef_):
        efek_per_penggandaan = None
        if "=" in nama:
            fitur, kategori = nama.split("=", 1)
            keterangan = f"{kategori} dibanding {pp.kategori_acuan_[fitur]} (acuan)"
            jenis = "kategori"
        elif nama in std_asli and nama.endswith("_log2"):
            efek_per_penggandaan = float((np.exp(w / std_asli[nama]) - 1) * 100)
            keterangan = f"per 2x lipat kapasitas: {efek_per_penggandaan:+.1f}%"
            jenis = "numerik"
        elif nama in std_asli:
            keterangan = f"per +1 std (= {std_asli[nama]:.3g} satuan asli)"
            jenis = "numerik"
        elif nama in pp.fitur_biner:
            keterangan = "ada dibanding tidak ada"
            jenis = "biner"
        else:
            keterangan = "suku polinomial (kuadrat/interaksi fitur terstandar)"
            jenis = "polinomial"
        hasil.append(
            {
                "fitur": nama,
                "jenis": jenis,
                "koefisien": float(w),
                "efek_persen": float((np.exp(w) - 1) * 100),
                "efek_per_penggandaan_persen": efek_per_penggandaan,
                "keterangan": keterangan,
            }
        )
    hasil.sort(key=lambda h: -h["koefisien"])
    return hasil


# ===========================================================================
# BAGIAN 5 — VERIFIKASI MANUAL vs SKLEARN & MULTIKOLINEARITAS
# ===========================================================================
def bandingkan_koefisien(regresor_a, regresor_b, X: np.ndarray) -> dict:
    """Selisih maksimum bobot & prediksi antara dua regresor yang sudah dilatih."""
    return {
        "selisih_maks_koefisien": float(
            np.max(np.abs(np.asarray(regresor_a.coef_) - np.asarray(regresor_b.coef_)))
        ),
        "selisih_intersep": float(abs(regresor_a.intercept_ - regresor_b.intercept_)),
        "selisih_maks_prediksi_log": float(np.max(np.abs(regresor_a.predict(X) - regresor_b.predict(X)))),
        "jumlah_koefisien": int(len(regresor_a.coef_)),
    }


def hitung_vif(X: np.ndarray, nama: list[str]) -> list[dict]:
    """Variance Inflation Factor tiap fitur.

    VIF_j = 1 / (1 - R²_j), dengan R²_j adalah R² ketika fitur j diregresikan
    terhadap semua fitur lain (memakai Normal Equation manual kita).
    VIF = 1  -> tidak berkorelasi dengan fitur lain.
    VIF > 10 -> multikolinearitas tinggi: koefisien fitur itu tidak stabil
               dan sulit ditafsirkan.
    """
    from ml.model import RegresiLinearNormalEquation

    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    hasil = []
    for j in range(Z.shape[1]):
        lain = np.delete(Z, j, axis=1)
        r2_j = r2(Z[:, j], RegresiLinearNormalEquation().fit(lain, Z[:, j]).predict(lain))
        hasil.append(
            {"fitur": nama[j], "r2_terhadap_fitur_lain": float(r2_j), "vif": float(1.0 / max(1.0 - r2_j, 1e-12))}
        )
    return hasil


def nilai_eigen(X: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Nilai eigen matriks (XbᵀXb + αI'), diurutkan dari kecil ke besar."""
    from ml.model import tambah_kolom_bias

    Xb = tambah_kolom_bias(X)
    I_aksen = np.eye(Xb.shape[1])
    I_aksen[0, 0] = 0.0
    return np.linalg.eigvalsh(Xb.T @ Xb + alpha * I_aksen)


def condition_number(X: np.ndarray, alpha: float = 0.0) -> float | None:
    """Condition number = nilai eigen terbesar / terkecil dari (XbᵀXb + αI').

    Makin besar, makin "rapuh" inversnya: perubahan kecil pada data bisa
    mengubah bobot secara besar. Ridge menambah alpha ke diagonal sehingga
    nilai eigen terkecil naik dan condition number turun.

    Mengembalikan None jika matriks singular (nilai eigen terkecil praktis
    nol, relatif terhadap yang terbesar) — condition number-nya tak hingga
    dan invers biasa tidak ada.
    """
    eigen = nilai_eigen(X, alpha)
    if eigen.min() <= eigen.max() * 1e-12:
        return None
    return float(eigen.max() / eigen.min())
