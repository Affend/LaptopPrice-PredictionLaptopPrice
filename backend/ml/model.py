"""
model.py — Implementasi model regresi.

Semua model di file ini mengikuti antarmuka yang sama dengan scikit-learn:
    model.fit(X, y)        -> mempelajari bobot dari data latih
    model.predict(X)       -> menghasilkan prediksi
    model.intercept_       -> bias / intersep (w0)
    model.coef_            -> bobot tiap fitur (w1..wp)

Dengan antarmuka yang sama, model manual dan model scikit-learn bisa
dievaluasi dengan kode yang persis sama, sehingga perbandingannya adil.

Notasi yang dipakai di komentar:
    n  = jumlah sampel (baris data)
    p  = jumlah fitur (kolom X)
    X  = matriks fitur berukuran (n, p)
    Xb = X yang sudah ditambah kolom angka 1 di depan -> ukuran (n, p+1).
         Kolom angka 1 ini dikalikan dengan w0, sehingga w0 berperan
         sebagai intersep.
    w  = vektor bobot [w0, w1, ..., wp]
    y  = vektor target berukuran (n,)

Model regresi linear berganda:
    y_hat = w0 + w1*x1 + w2*x2 + ... + wp*xp = Xb @ w
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def tambah_kolom_bias(X: np.ndarray) -> np.ndarray:
    """Menambahkan kolom berisi angka 1 di sisi kiri matriks X.

    Contoh: X = [[2, 3],     menjadi     Xb = [[1, 2, 3],
                 [4, 5]]                       [1, 4, 5]]
    """
    X = np.asarray(X, dtype=float)
    kolom_satu = np.ones((X.shape[0], 1))
    return np.hstack([kolom_satu, X])


class _ModelLinearDasar:
    """Kelas induk: menyimpan bobot dan menyediakan fungsi predict().

    Semua model linear (Normal Equation, Gradient Descent, Ridge) memprediksi
    dengan cara yang sama, yaitu y_hat = w0 + X @ [w1..wp]. Yang berbeda
    hanyalah cara mencari bobot w di dalam fit().
    """

    intercept_: float
    coef_: np.ndarray

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def _simpan_bobot(self, w: np.ndarray) -> None:
        # w[0] adalah intersep, sisanya bobot tiap fitur
        self.intercept_ = float(w[0])
        self.coef_ = np.asarray(w[1:], dtype=float)


# ---------------------------------------------------------------------------
# 1. Regresi Linear Berganda — Normal Equation (solusi tertutup)
# ---------------------------------------------------------------------------
class RegresiLinearNormalEquation(_ModelLinearDasar):
    """Mencari bobot terbaik secara langsung (satu langkah hitung).

    Tujuan regresi linear adalah meminimalkan Sum of Squared Errors:
        J(w) = sum (y - Xb @ w)^2

    Turunan J terhadap w disamakan dengan nol, sehingga diperoleh
    "persamaan normal":
        Xb^T Xb w = Xb^T y
        w = (Xb^T Xb)^(-1) Xb^T y

    Kenapa memakai pseudo-inverse (np.linalg.pinv), bukan invers biasa?
    Jika ada fitur yang saling bergantung linear (multikolinearitas sempurna),
    matriks Xb^T Xb menjadi singular dan tidak punya invers. Pseudo-inverse
    (dihitung lewat SVD) tetap memberikan solusi dengan norma bobot terkecil,
    sehingga perhitungan tidak gagal.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresiLinearNormalEquation":
        Xb = tambah_kolom_bias(X)
        y = np.asarray(y, dtype=float)

        XtX = Xb.T @ Xb           # ukuran (p+1, p+1)
        Xty = Xb.T @ y            # ukuran (p+1,)
        w = np.linalg.pinv(XtX) @ Xty

        self._simpan_bobot(w)
        return self


# ---------------------------------------------------------------------------
# 2. Regresi Linear Berganda — Gradient Descent (iteratif)
# ---------------------------------------------------------------------------
class RegresiLinearGradientDescent(_ModelLinearDasar):
    """Mencari bobot secara bertahap dengan menuruni "lembah" fungsi loss.

    Fungsi loss yang dipakai adalah Mean Squared Error (MSE):
        J(w) = (1/n) * sum (Xb @ w - y)^2

    Gradien (arah kenaikan tercuram) dari J terhadap w:
        dJ/dw = (2/n) * Xb^T (Xb @ w - y)

    Setiap iterasi, bobot digeser berlawanan arah gradien:
        w_baru = w_lama - learning_rate * dJ/dw

    Catatan penting:
    - Fitur harus sudah distandardisasi (rata-rata 0, standar deviasi 1).
      Jika skala fitur berbeda jauh (mis. RAM 4..64 vs SSD 0..1000),
      lembah loss menjadi sangat lonjong dan GD lambat atau tidak konvergen.
    - learning_rate terlalu besar -> loss naik / meledak (divergen).
      learning_rate terlalu kecil -> butuh sangat banyak iterasi.
    - Riwayat loss tiap iterasi disimpan di `riwayat_loss_` untuk digambar
      sebagai kurva loss (bukti bahwa algoritma konvergen).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,
        n_iterasi: int = 5000,
        toleransi: float | None = 1e-12,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterasi = n_iterasi
        # Iterasi dihentikan lebih awal jika penurunan loss < toleransi
        self.toleransi = toleransi

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresiLinearGradientDescent":
        Xb = tambah_kolom_bias(X)
        y = np.asarray(y, dtype=float)
        n, p_plus_1 = Xb.shape

        w = np.zeros(p_plus_1)    # titik awal: semua bobot nol
        riwayat_loss: list[float] = []

        # errstate: kita sendiri yang menangani kasus loss meledak (lihat di
        # bawah), jadi peringatan overflow bawaan NumPy tidak perlu ditampilkan.
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(self.n_iterasi):
                error = Xb @ w - y                    # selisih prediksi - aktual
                loss = float(np.mean(error ** 2))     # MSE saat ini

                # Berhenti jika loss meledak (learning rate terlalu besar)
                if not np.isfinite(loss):
                    raise ValueError(
                        "Gradient Descent divergen (loss tak hingga). "
                        "Coba perkecil learning_rate."
                    )
                riwayat_loss.append(loss)

                # Berhenti jika loss sudah tidak turun berarti (konvergen)
                if (
                    self.toleransi is not None
                    and len(riwayat_loss) > 1
                    and abs(riwayat_loss[-2] - loss) < self.toleransi
                ):
                    break

                gradien = (2.0 / n) * (Xb.T @ error)
                w = w - self.learning_rate * gradien

        self.riwayat_loss_ = riwayat_loss
        self.n_iterasi_dijalankan_ = len(riwayat_loss)
        self._simpan_bobot(w)
        return self


# ---------------------------------------------------------------------------
# 3. Ridge Regression manual (regresi linear + penalti L2)
# ---------------------------------------------------------------------------
class RegresiRidgeManual(_ModelLinearDasar):
    """Regresi linear dengan penalti ukuran bobot (regularisasi L2).

    Fungsi yang diminimalkan:
        J(w) = sum (y - Xb @ w)^2 + alpha * sum_{j=1..p} wj^2

    Solusi tertutupnya:
        w = (Xb^T Xb + alpha * I')^(-1) Xb^T y

    I' adalah matriks identitas yang elemen [0, 0]-nya dibuat 0, karena
    intersep w0 TIDAK ikut dipenalti (intersep hanya menggeser rata-rata,
    bukan sumber overfitting). Ini sama dengan perilaku sklearn.linear_model.Ridge.

    Kenapa Ridge membantu saat multikolinearitas?
    Jika dua fitur sangat berkorelasi (mis. lebar dan tinggi resolusi),
    Xb^T Xb hampir singular sehingga bobot OLS bisa sangat besar dan
    berlawanan tanda. Penambahan alpha pada diagonal membuat matriks lebih
    "sehat" (condition number turun) dan bobot menjadi lebih kecil & stabil.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegresiRidgeManual":
        Xb = tambah_kolom_bias(X)
        y = np.asarray(y, dtype=float)

        I_aksen = np.eye(Xb.shape[1])
        I_aksen[0, 0] = 0.0                       # intersep tidak dipenalti

        A = Xb.T @ Xb + self.alpha * I_aksen
        b = Xb.T @ y
        # np.linalg.solve(A, b) menghitung A^(-1) b tanpa membentuk invers
        # secara eksplisit (lebih stabil secara numerik). Jika A singular
        # (alpha = 0 dan ada kolinearitas sempurna), pakai pseudo-inverse.
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(A) @ b

        self._simpan_bobot(w)
        return self


# ---------------------------------------------------------------------------
# 4. Baseline: selalu menebak rata-rata harga
# ---------------------------------------------------------------------------
class BaselineRataRata:
    """Model paling sederhana: apa pun inputnya, tebak rata-rata harga.

    Gunanya sebagai batas bawah. Model regresi baru dianggap berguna jika
    error-nya jauh lebih kecil dari baseline ini. R^2 baseline pada data
    latih bernilai 0 (definisi R^2 = 1 - SSE_model / SSE_rata-rata).

    Catatan skala: pipeline memodelkan y = ln(harga). Agar setelah
    dikembalikan dengan exp() tebakannya tepat sama dengan rata-rata harga
    dalam euro, konstanta disimpan sebagai ln(rata-rata harga), bukan
    rata-rata dari ln(harga). (Rata-rata dari ln(harga) jika di-exp()
    menghasilkan rata-rata geometrik, yang lebih kecil dari rata-rata biasa.)
    """

    def fit(self, X: np.ndarray, y_log: np.ndarray) -> "BaselineRataRata":
        harga_euro = np.exp(np.asarray(y_log, dtype=float))
        self.rata_rata_harga_ = float(np.mean(harga_euro))
        self.intercept_ = float(np.log(self.rata_rata_harga_))
        self.coef_ = np.zeros(np.asarray(X).shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(X).shape[0], self.intercept_)


# ---------------------------------------------------------------------------
# 5. Pembungkus: Preprocessor + regresor = satu model utuh
# ---------------------------------------------------------------------------
class ModelHargaLaptop:
    """Menggabungkan Preprocessor dan regresor menjadi satu objek.

    Dengan digabung, preprocessing selalu di-fit ulang setiap kali model
    di-fit. Ini penting di Cross Validation: di setiap fold, rata-rata/std
    dan daftar kategori dihitung hanya dari data latih fold tersebut.
    Objek inilah yang disimpan ke file .joblib dan dimuat oleh API.
    """

    def __init__(self, nama: str, regresor, preprocessor, hyperparameter: dict | None = None):
        self.nama = nama
        self.regresor = regresor
        self.preprocessor = preprocessor
        self.hyperparameter = hyperparameter or {}

    def fit(self, df_fitur, y_log: np.ndarray) -> "ModelHargaLaptop":
        X = self.preprocessor.fit_transform(df_fitur)
        self.regresor.fit(X, y_log)
        return self

    def predict_log(self, df_fitur) -> np.ndarray:
        """Prediksi pada skala ln(harga)."""
        return self.regresor.predict(self.preprocessor.transform(df_fitur))

    def predict_euro(self, df_fitur) -> np.ndarray:
        """Prediksi harga dalam euro: exp(prediksi ln harga)."""
        return np.exp(self.predict_log(df_fitur))

    def kontribusi(self, df_fitur_satu_baris) -> dict:
        """Menguraikan satu prediksi menjadi sumbangan tiap fitur.

        Karena modelnya linear:  ln(harga) = w0 + sum_j (wj * xj)
        maka sumbangan kolom j terhadap ln(harga) adalah wj * xj.
        Kolom-kolom yang berasal dari fitur yang sama (mis. semua kolom
        one-hot "Company=...") dijumlahkan.

        Titik acuan (w0) adalah laptop "acuan": semua fitur numerik bernilai
        rata-rata data latih, fitur biner = 0, dan semua kategori = kategori
        acuan (yang paling sering muncul). Sumbangan sebesar s pada skala log
        berarti harga dikali exp(s), yaitu berubah (exp(s) - 1) * 100 persen.
        """
        x = self.preprocessor.transform(df_fitur_satu_baris)[0]
        sumbangan_kolom = np.asarray(self.regresor.coef_) * x
        per_fitur: dict[str, float] = {}
        for nama_kolom, s in zip(self.preprocessor.nama_fitur_, sumbangan_kolom):
            asal = self.preprocessor.fitur_asal(nama_kolom)
            per_fitur[asal] = per_fitur.get(asal, 0.0) + float(s)
        return {
            "intersep_log": float(self.regresor.intercept_),
            "harga_acuan_euro": float(np.exp(self.regresor.intercept_)),
            "fitur": [
                {"fitur": f, "kontribusi_log": s, "efek_persen": (np.exp(s) - 1) * 100}
                for f, s in sorted(per_fitur.items(), key=lambda kv: -abs(kv[1]))
            ],
        }


# ---------------------------------------------------------------------------
# 6. Daftar model yang dibandingkan
# ---------------------------------------------------------------------------
# Setiap entri: label tampilan, keterangan singkat, dan fungsi pembuat model.
# Fungsi pembuat dipakai agar setiap fold CV mendapat model BARU yang belum
# pernah dilatih (tidak membawa sisa bobot dari fold lain).
def _preprocessor(polinomial: bool = False):
    # import di dalam fungsi agar model.py tetap bisa diuji tanpa pandas
    from ml.preprocessing import FITUR_POLINOMIAL, Preprocessor

    return Preprocessor(fitur_polinomial=FITUR_POLINOMIAL if polinomial else None)


def _sklearn_linear():
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _sklearn_ridge(alpha: float):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha)


SPESIFIKASI_MODEL: dict[str, dict] = {
    "baseline_rata_rata": {
        "label": "Baseline (rata-rata harga)",
        "keterangan": "Selalu menebak rata-rata harga data latih. Batas bawah pembanding.",
        "buat": lambda h: (BaselineRataRata(), _preprocessor()),
    },
    "ols_normal_equation": {
        "label": "Regresi Linear — Normal Equation (manual)",
        "keterangan": "w = pinv(XᵀX)·Xᵀy, dihitung dengan NumPy.",
        "buat": lambda h: (RegresiLinearNormalEquation(), _preprocessor()),
    },
    "ols_gradient_descent": {
        "label": "Regresi Linear — Gradient Descent (manual)",
        "keterangan": "Bobot dicari bertahap dengan menurunkan MSE.",
        "buat": lambda h: (
            RegresiLinearGradientDescent(h["learning_rate"], h["n_iterasi"], h["toleransi"]),
            _preprocessor(),
        ),
    },
    "ols_sklearn": {
        "label": "Regresi Linear — scikit-learn",
        "keterangan": "sklearn.linear_model.LinearRegression sebagai pembanding.",
        "buat": lambda h: (_sklearn_linear(), _preprocessor()),
    },
    "ridge_manual": {
        "label": "Ridge Regression (manual)",
        "keterangan": "w = (XᵀX + αI')⁻¹Xᵀy; α dipilih lewat 5-Fold CV.",
        "buat": lambda h: (RegresiRidgeManual(h["alpha"]), _preprocessor()),
    },
    "ridge_sklearn": {
        "label": "Ridge Regression — scikit-learn",
        "keterangan": "sklearn.linear_model.Ridge dengan α yang sama.",
        "buat": lambda h: (_sklearn_ridge(h["alpha"]), _preprocessor()),
    },
    "polinomial_d2": {
        "label": "Regresi Polinomial derajat 2 (manual)",
        "keterangan": "Kuadrat & interaksi RAM, PPI, SSD, berat, GHz + fitur linear lain; Normal Equation.",
        "buat": lambda h: (RegresiLinearNormalEquation(), _preprocessor(polinomial=True)),
    },
}


def buat_model(nama: str, hyperparameter: dict | None = None) -> ModelHargaLaptop:
    """Membuat model baru (belum dilatih) berdasarkan namanya."""
    hyperparameter = dict(hyperparameter or {})
    regresor, preprocessor = SPESIFIKASI_MODEL[nama]["buat"](hyperparameter)
    return ModelHargaLaptop(nama, regresor, preprocessor, hyperparameter)
