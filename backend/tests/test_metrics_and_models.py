"""
Pengujian metrik manual dan model regresi manual.

Ide pengujiannya: hasil hitungan manual (NumPy) harus SAMA dengan hasil
library scikit-learn yang sudah teruji luas. Jika sama, rumus kita benar.
Data yang dipakai di sini adalah data sintetis (buatan) yang bobot aslinya
kita ketahui, sehingga kita juga bisa mengecek apakah model menemukan
bobot tersebut.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from ml.evaluation import hitung_metrik_sklearn, hitung_semua_metrik, mae, mape, r2, rmse, verifikasi_metrik
from ml.model import (
    BaselineRataRata,
    RegresiLinearGradientDescent,
    RegresiLinearNormalEquation,
    RegresiRidgeManual,
)


@pytest.fixture
def data_sintetis():
    """y = 3 + 2*x1 - 1*x2 + 0.5*x3 + noise kecil, fitur sudah standar."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    w_asli = np.array([2.0, -1.0, 0.5])
    y = 3.0 + X @ w_asli + rng.normal(scale=0.1, size=200)
    return X, y, w_asli


# --------------------------- Metrik ---------------------------------------
def test_metrik_hitungan_tangan():
    """Contoh kecil yang bisa dihitung manual di kertas."""
    y = np.array([100.0, 200.0, 300.0])
    y_hat = np.array([110.0, 190.0, 330.0])
    # error = [-10, 10, -30]
    assert mae(y, y_hat) == pytest.approx((10 + 10 + 30) / 3)
    assert rmse(y, y_hat) == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert mape(y, y_hat) == pytest.approx((10 / 100 + 10 / 200 + 30 / 300) / 3 * 100)
    # SST = (100-200)^2 + 0 + (300-200)^2 = 20000, SSE = 1100
    assert r2(y, y_hat) == pytest.approx(1 - 1100 / 20000)


def test_metrik_manual_sama_dengan_sklearn():
    rng = np.random.default_rng(1)
    y = rng.uniform(200, 3000, size=500)
    y_hat = y * rng.normal(1.0, 0.2, size=500)
    manual = hitung_semua_metrik(y, y_hat)
    pembanding = hitung_metrik_sklearn(y, y_hat)
    for nama in manual:
        assert manual[nama] == pytest.approx(pembanding[nama], rel=1e-10)
    assert all(baris["sama"] for baris in verifikasi_metrik(y, y_hat))


def test_r2_prediksi_rata_rata_bernilai_nol():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(y, np.full(4, y.mean())) == pytest.approx(0.0)


# --------------------------- Model -----------------------------------------
def test_normal_equation_sama_dengan_sklearn(data_sintetis):
    X, y, w_asli = data_sintetis
    manual = RegresiLinearNormalEquation().fit(X, y)
    sk = LinearRegression().fit(X, y)
    assert manual.intercept_ == pytest.approx(sk.intercept_, abs=1e-8)
    np.testing.assert_allclose(manual.coef_, sk.coef_, atol=1e-8)
    # bobot yang ditemukan mendekati bobot asli
    np.testing.assert_allclose(manual.coef_, w_asli, atol=0.05)


def test_normal_equation_tahan_kolinearitas_sempurna(data_sintetis):
    """Kolom ke-4 = 2 x kolom ke-1 -> X^T X singular, pinv tetap jalan."""
    X, y, _ = data_sintetis
    X_kolinear = np.hstack([X, 2 * X[:, [0]]])
    model = RegresiLinearNormalEquation().fit(X_kolinear, y)
    sk = LinearRegression().fit(X_kolinear, y)
    np.testing.assert_allclose(model.predict(X_kolinear), sk.predict(X_kolinear), atol=1e-6)


def test_gradient_descent_konvergen_ke_normal_equation(data_sintetis):
    X, y, _ = data_sintetis
    gd = RegresiLinearGradientDescent(learning_rate=0.1, n_iterasi=5000).fit(X, y)
    ne = RegresiLinearNormalEquation().fit(X, y)
    np.testing.assert_allclose(gd.coef_, ne.coef_, atol=1e-4)
    assert gd.intercept_ == pytest.approx(ne.intercept_, abs=1e-4)
    # loss harus turun dari awal ke akhir
    assert gd.riwayat_loss_[-1] < gd.riwayat_loss_[0]


def test_gradient_descent_divergen_memberi_pesan(data_sintetis):
    X, y, _ = data_sintetis
    with pytest.raises(ValueError, match="divergen"):
        RegresiLinearGradientDescent(learning_rate=10.0, n_iterasi=2000).fit(X, y)


@pytest.mark.parametrize("alpha", [0.1, 1.0, 10.0, 100.0])
def test_ridge_manual_sama_dengan_sklearn(data_sintetis, alpha):
    X, y, _ = data_sintetis
    manual = RegresiRidgeManual(alpha=alpha).fit(X, y)
    sk = Ridge(alpha=alpha).fit(X, y)
    assert manual.intercept_ == pytest.approx(sk.intercept_, abs=1e-8)
    np.testing.assert_allclose(manual.coef_, sk.coef_, atol=1e-8)


def test_baseline_menebak_rata_rata_harga():
    harga = np.array([500.0, 1000.0, 1500.0])
    model = BaselineRataRata().fit(np.zeros((3, 2)), np.log(harga))
    prediksi_euro = np.exp(model.predict(np.zeros((5, 2))))
    np.testing.assert_allclose(prediksi_euro, 1000.0)
