"""
Pengujian preprocessing & feature engineering.

Yang diuji:
1. Setiap fungsi parsing menghasilkan angka yang benar untuk format yang
   benar-benar ada di dataset (termasuk format "aneh" seperti dua spasi
   setelah '+' dan "1.0TB Hybrid").
2. Preprocessor tidak membocorkan informasi data uji (fit hanya di latih).
3. Fitur dari form website identik dengan fitur dari CSV.
"""

import numpy as np
import pandas as pd
import pytest

from ml.config import ambil_pengaturan
from ml.model import buat_model
from ml.preprocessing import (
    FITUR_KATEGORI,
    FITUR_NUMERIK,
    Preprocessor,
    bersihkan_data,
    ekstrak_fitur,
    fitur_dari_input,
    hitung_ppi,
    kategori_cpu,
    ke_angka,
    merek_gpu,
    muat_dataset,
    parse_memori,
    parse_resolusi,
    sederhanakan_os,
    target_log,
)


@pytest.fixture(scope="module")
def data_asli():
    df, dibuang = bersihkan_data(muat_dataset(ambil_pengaturan().path_dataset))
    return df, dibuang, ekstrak_fitur(df)


# --------------------------- Fungsi parsing --------------------------------
@pytest.mark.parametrize(
    "teks, ssd, hdd, flash",
    [
        ("256GB SSD", 256, 0, 0),
        ("1TB HDD", 0, 1000, 0),
        ("128GB SSD +  1TB HDD", 128, 1000, 0),  # dua spasi, persis seperti di CSV
        ("1.0TB Hybrid", 0, 1000, 0),  # Hybrid dihitung sebagai HDD
        ("64GB Flash Storage +  1TB HDD", 0, 1000, 64),
        ("256GB SSD +  256GB SSD", 512, 0, 0),  # dua drive sejenis dijumlah
        ("512GB SSD +  1.0TB Hybrid", 512, 1000, 0),
        ("508GB Hybrid", 0, 508, 0),
    ],
)
def test_parse_memori(teks, ssd, hdd, flash):
    assert parse_memori(teks) == {"SSD_GB": ssd, "HDD_GB": hdd, "Flash_GB": flash}


def test_parse_memori_format_tidak_dikenal():
    with pytest.raises(ValueError):
        parse_memori("256GB Optane")


@pytest.mark.parametrize(
    "teks, lebar, tinggi, sentuh, ips",
    [
        ("1366x768", 1366, 768, 0, 0),
        ("Full HD 1920x1080", 1920, 1080, 0, 0),
        ("IPS Panel Full HD / Touchscreen 1920x1080", 1920, 1080, 1, 1),
        ("Touchscreen / 4K Ultra HD 3840x2160", 3840, 2160, 1, 0),
        ("IPS Panel Retina Display 2560x1600", 2560, 1600, 0, 1),
    ],
)
def test_parse_resolusi(teks, lebar, tinggi, sentuh, ips):
    assert parse_resolusi(teks) == {"Lebar_px": lebar, "Tinggi_px": tinggi, "Touchscreen": sentuh, "IPS": ips}


def test_hitung_ppi():
    # diagonal 1920x1080 = sqrt(1920^2 + 1080^2) = 2202.9 piksel
    assert hitung_ppi(1920, 1080, 15.6) == pytest.approx(2202.907 / 15.6, rel=1e-4)


@pytest.mark.parametrize(
    "perusahaan, tipe, hasil",
    [
        ("Intel", "Core i5 7200U", "Intel Core i5"),
        ("Intel", "Core i7", "Intel Core i7"),
        ("Intel", "Core i3 6006U", "Intel Core i3"),
        ("Intel", "Core M m3-7Y30", "Intel Lainnya"),  # "Core M" bukan Core i
        ("Intel", "Celeron Dual Core N3350", "Intel Lainnya"),
        ("Intel", "Xeon E3-1505M V6", "Intel Lainnya"),
        ("AMD", "Ryzen 1700", "AMD"),
        ("Samsung", "Cortex A72&A53", "Lainnya"),
    ],
)
def test_kategori_cpu(perusahaan, tipe, hasil):
    assert kategori_cpu(perusahaan, tipe) == hasil


def test_merek_gpu_dan_os():
    assert merek_gpu("Nvidia") == "Nvidia"
    assert merek_gpu("ARM") == "Lainnya"
    assert sederhanakan_os("Windows 10 S") == "Windows"
    assert sederhanakan_os("Mac OS X") == "macOS"
    assert sederhanakan_os("No OS") == "Tanpa OS"
    assert sederhanakan_os("Android") == "Chrome OS/Android"


def test_ke_angka_menerima_angka_dan_teks():
    assert ke_angka(8) == 8.0
    assert ke_angka("8GB", "gb") == 8.0
    assert ke_angka("1.37kg", "kg") == 1.37


# --------------------------- Dataset asli ----------------------------------
def test_dataset_asli_terbaca_lengkap(data_asli):
    df, dibuang, F = data_asli
    assert len(df) == 1274  # 1275 baris mentah - 1 laptop ARM
    assert len(dibuang) == 1 and dibuang[0]["product"] == "Chromebook Plus"
    assert not F.isna().any().any()
    assert set(F["CPU_Kategori"]) <= {"Intel Core i7", "Intel Core i5", "Intel Core i3", "Intel Lainnya", "AMD"}
    assert np.all(target_log(df) == np.log(df["Price_euros"]))


# --------------------------- Preprocessor ----------------------------------
def test_preprocessor_fit_hanya_dari_data_latih(data_asli):
    _, _, F = data_asli
    latih, uji = F.iloc[:800], F.iloc[800:]
    pp = Preprocessor().fit(latih)
    # rata-rata & std berasal dari data latih saja
    np.testing.assert_allclose(pp.rata_rata_, latih[FITUR_NUMERIK].mean().to_numpy())
    np.testing.assert_allclose(pp.std_, latih[FITUR_NUMERIK].std(ddof=0).to_numpy())
    # transform data uji tidak mengubah parameter yang sudah dipelajari
    sebelum = pp.rata_rata_.copy()
    pp.transform(uji)
    np.testing.assert_array_equal(pp.rata_rata_, sebelum)
    # kolom numerik data latih terstandar: rata-rata 0, std 1
    X = pp.transform(latih)[:, : len(FITUR_NUMERIK)]
    np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-10)


def _tabel_kecil(company, os_):
    n = len(company)
    return pd.DataFrame(
        {
            "Inches": np.linspace(13, 17, n), "Ram_log2": np.linspace(2, 5, n), "Weight": np.linspace(1, 3, n),
            "CPU_GHz": np.linspace(1.5, 3, n), "PPI": np.linspace(100, 200, n), "SSD_log2": np.linspace(0, 9, n),
            "HDD_log2": np.zeros(n), "Flash_log2": np.zeros(n), "Touchscreen": 0, "IPS": 0,
            "Company": company, "TypeName": "Notebook", "CPU_Kategori": "Intel Core i5",
            "GPU_Merek": "Intel", "OS": os_,
        }
    )


def test_one_hot_drop_first_dan_kategori_langka():
    company = ["Dell"] * 10 + ["HP"] * 6 + ["Razer"] * 2 + ["Vero"] * 3
    pp = Preprocessor(min_frekuensi=5).fit(_tabel_kecil(company, "Windows"))
    # Dell = acuan (terbanyak) -> tidak punya kolom; Razer & Vero (< 5) -> Lainnya
    assert pp.kategori_acuan_["Company"] == "Dell"
    assert "Company=Dell" not in pp.nama_fitur_
    assert "Company=HP" in pp.nama_fitur_ and "Company=Lainnya" in pp.nama_fitur_
    assert "Company=Razer" not in pp.nama_fitur_


def test_kategori_tidak_dikenal_tidak_error():
    company = ["Dell"] * 10 + ["HP"] * 6
    pp = Preprocessor(min_frekuensi=5).fit(_tabel_kecil(company, "Windows"))
    X = pp.transform(_tabel_kecil(["MerekBaru"], "Windows"))
    kolom_hp = pp.nama_fitur_.index("Company=HP")
    assert X[0, kolom_hp] == 0  # diperlakukan seperti kategori acuan


def test_kolom_one_hot_identik_dibuang():
    """Seperti Apple ⇔ macOS di dataset asli: dua kolom selalu bernilai sama."""
    company = ["Dell"] * 10 + ["Apple"] * 6
    os_ = ["Windows"] * 10 + ["macOS"] * 6
    pp = Preprocessor(min_frekuensi=5).fit(_tabel_kecil(company, os_))
    assert pp.kolom_duplikat_ == [{"dibuang": "OS=macOS", "identik_dengan": "Company=Apple"}]
    assert "OS=macOS" not in pp.nama_fitur_


def test_apple_macos_terdeteksi_di_dataset_asli(data_asli):
    _, _, F = data_asli
    pp = Preprocessor().fit(F)
    assert {"dibuang": "OS=macOS", "identik_dengan": "Company=Apple"} in pp.kolom_duplikat_
    X = pp.transform(F)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    assert np.linalg.matrix_rank(Xb) == Xb.shape[1]  # tidak ada kolinearitas sempurna


# --------------------------- Konsistensi API vs training --------------------
def test_fitur_dari_input_sama_dengan_ekstrak_fitur(data_asli):
    df, _, F = data_asli
    kolom_model = FITUR_NUMERIK + ["Touchscreen", "IPS"] + FITUR_KATEGORI
    for i in [0, 3, 100, 777]:
        baris = F.iloc[i]
        dari_form = fitur_dari_input(baris.to_dict())
        pd.testing.assert_series_equal(
            dari_form[kolom_model].iloc[0], baris[kolom_model], check_names=False, check_dtype=False
        )


def test_kontribusi_fitur_menjumlah_ke_prediksi(data_asli):
    """intersep + jumlah kontribusi semua fitur = prediksi ln(harga)."""
    df, _, F = data_asli
    y = target_log(df)
    for nama in ("ols_normal_equation", "polinomial_d2"):
        model = buat_model(nama).fit(F, y)
        satu = F.iloc[[5]]
        k = model.kontribusi(satu)
        total = k["intersep_log"] + sum(f["kontribusi_log"] for f in k["fitur"])
        assert total == pytest.approx(model.predict_log(satu)[0], abs=1e-10)
