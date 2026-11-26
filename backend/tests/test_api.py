"""
Pengujian endpoint API (terutama POST /api/predict).

Aplikasi dijalankan dengan database SQLite sementara yang berisi satu hasil
training, lewat TestClient (tanpa perlu menyalakan server sungguhan).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import buat_app
from db import repository
from ml.config import Z_95
from ml.preprocessing import bersihkan_data, ekstrak_fitur, muat_dataset
from ml.config import ambil_pengaturan

INPUT_CONTOH = {
    "company": "Dell", "type_name": "Notebook", "inches": 15.6, "ram_gb": 8, "berat_kg": 2.04,
    "cpu_kategori": "Intel Core i5", "cpu_ghz": 2.5, "lebar_px": 1920, "tinggi_px": 1080,
    "touchscreen": False, "ips": False, "ssd_gb": 256, "hdd_gb": 0, "flash_gb": 0,
    "gpu_merek": "Intel", "os": "Windows",
}


@pytest.fixture
def klien(db_sementara, hasil_training):
    pembuat_session, folder_model = db_sementara
    hasil, model_terlatih = hasil_training
    with pembuat_session() as s:
        repository.simpan_hasil_training(s, hasil, model_terlatih, folder_model)
    with TestClient(buat_app(pembuat_session=pembuat_session)) as c:
        yield c, model_terlatih


@pytest.fixture
def klien_tanpa_model(db_sementara):
    pembuat_session, _ = db_sementara
    with TestClient(buat_app(pembuat_session=pembuat_session)) as c:
        yield c


# --------------------------- Endpoint umum ---------------------------------
def test_health(klien):
    c, _ = klien
    r = c.get("/api/health").json()
    assert r["status"] == "ok" and r["database"] == "ok"
    assert r["model_aktif"]["nama_model"] == "ols_normal_equation"


def test_dataset_summary_dan_eda(klien):
    c, _ = klien
    ringkas = c.get("/api/dataset/summary").json()
    assert ringkas["jumlah_baris_mentah"] == 1275 and ringkas["jumlah_baris_dipakai"] == 1274
    eda = c.get("/api/dataset/eda").json()
    assert sum(b["frekuensi"] for b in eda["distribusi_harga"]) == 1274
    assert len(eda["korelasi"]["nilai"]) == len(eda["korelasi"]["kolom"])


def test_options_dari_dataset(klien):
    c, _ = klien
    r = c.get("/api/options").json()
    assert r["company"][0]["nilai"] == "Dell"  # merek terbanyak di dataset
    assert {o["nilai"] for o in r["cpu_kategori"]} == {"Intel Core i7", "Intel Core i5", "Intel Lainnya", "Intel Core i3", "AMD"}
    assert r["penyesuaian_harga"]["kurs_eur_idr"] > 0
    assert r["kategori_model"]["Company"][0] == "Dell"


def test_options_model_cpu_gpu_dan_berat(klien):
    """Form cukup memilih model CPU/GPU; kategori & clock ikut dari dataset."""
    c, _ = klien
    r = c.get("/api/options").json()
    cpu = {d["nilai"]: d for d in r["cpu_model"]}
    assert cpu["Intel Core i5 7200U"]["kategori"] == "Intel Core i5"
    assert cpu["Intel Core i5 7200U"]["ghz"] == 2.5  # nilai clock yang paling sering untuk model ini
    assert sum(d["jumlah"] for d in r["cpu_model"]) == 1274
    assert "Intel Core i5 (model tidak disebut)" in cpu
    gpu = {d["nilai"]: d for d in r["gpu_model"]}
    assert gpu["Nvidia GeForce GTX 1050"]["merek"] == "Nvidia"
    # nilai default konsisten: kategori & clock sesuai model CPU default
    d = r["nilai_default"]
    assert cpu[d["cpu_model"]]["kategori"] == d["cpu_kategori"] and cpu[d["cpu_model"]]["ghz"] == d["cpu_ghz"]
    assert gpu[d["gpu_model"]]["merek"] == d["gpu_merek"]
    assert r["berat_tipikal"]["per_tipe_inci"]["Gaming|17.3"] > r["berat_tipikal"]["per_tipe_inci"]["Ultrabook|13.3"]
    assert {"lebar_px": 1920, "tinggi_px": 1080, "nama": "Full HD"}.items() <= r["resolusi"][0].items()


def test_cari_laptop(klien):
    c, _ = klien
    r = c.get("/api/dataset/laptop", params={"q": "dell xps"}).json()
    assert r["total_cocok"] > 0
    assert all("xps" in d["label"].lower() for d in r["data"])
    # laptop data uji tampil lebih dulu daripada data latih
    urutan = [d["set"] for d in c.get("/api/dataset/laptop", params={"q": "dell", "batas": 50}).json()["data"]]
    assert urutan == sorted(urutan, key=lambda s: s != "uji")
    # tanpa kata kunci: contoh laptop dari data UJI, terurut dari murah ke mahal
    contoh = c.get("/api/dataset/laptop").json()["data"]
    assert len(contoh) == 12 and all(d["set"] == "uji" for d in contoh)
    harga = [d["harga_euro"] for d in contoh]
    assert harga == sorted(harga)
    assert c.get("/api/dataset/laptop", params={"q": "tidakadalaptopini"}).json()["total_cocok"] == 0


def test_pembagian_latih_uji_sama_dengan_training(hasil_training):
    """Tanda 'uji' di daftar laptop harus sama persis dengan data uji saat training,
    dan harga yang ditampilkan harus sama dengan harga aktual data uji."""
    from app.layanan_dataset import daftar_laptop, muat_data_dataset

    hasil, _ = hasil_training
    daftar = daftar_laptop(muat_data_dataset(ambil_pengaturan().path_dataset))
    uji = [d for d in daftar if d["set"] == "uji"]
    assert len(uji) == hasil["dataset"]["jumlah_uji"] == 255
    harga_uji_training = sorted(t["aktual_euro"] for t in hasil["model"]["ols_normal_equation"]["residual"]["titik"])
    assert sorted(d["harga_euro"] for d in uji) == pytest.approx(harga_uji_training)


def test_prediksi_laptop_dari_dataset_sama_dengan_model(klien):
    """Isi form dari laptop dataset -> prediksi = prediksi model langsung untuk baris itu."""
    c, model_terlatih = klien
    laptop = c.get("/api/dataset/laptop", params={"q": "lenovo legion"}).json()["data"][0]
    d = c.post("/api/predict", json=laptop["input"]).json()
    df, _ = bersihkan_data(muat_dataset(ambil_pengaturan().path_dataset))
    langsung = model_terlatih["ols_normal_equation"].predict_euro(ekstrak_fitur(df).iloc[[laptop["id"]]])[0]
    assert d["harga_dataset_eur"] == pytest.approx(langsung, rel=1e-12)
    riwayat = c.get("/api/predictions/history").json()["data"][0]
    assert riwayat["input"]["cpu_model"] == laptop["input"]["cpu_model"]


# --------------------------- Prediksi -------------------------------------
def test_predict_berhasil_dan_tersimpan(klien):
    c, _ = klien
    r = c.post("/api/predict", json=INPUT_CONTOH)
    assert r.status_code == 200
    d = r.json()
    eur = d["harga_dataset_eur"]
    bawah, atas = d["rentang_dataset_eur"]
    assert bawah < eur < atas
    faktor = d["penyesuaian_harga"]["faktor_inflasi"]
    kurs = d["penyesuaian_harga"]["kurs_eur_idr"]
    assert d["harga_sekarang_eur"] == pytest.approx(eur * faktor)
    assert d["harga_idr"] == pytest.approx(eur * faktor * kurs)
    # rentang = exp(ln_prediksi ± 1,96 * RMSE_log)
    s = d["rentang"]["rmse_log_cv"]
    assert bawah == pytest.approx(np.exp(d["prediksi_log"] - Z_95 * s))
    # intersep + jumlah kontribusi = ln(harga prediksi)
    k = d["kontribusi"]
    assert k["intersep_log"] + sum(f["kontribusi_log"] for f in k["fitur"]) == pytest.approx(d["prediksi_log"])
    assert d["peringatan"] == []

    riwayat = c.get("/api/predictions/history").json()
    assert riwayat["total"] == 1
    assert riwayat["data"][0]["harga_prediksi_eur"] == pytest.approx(eur)
    assert riwayat["data"][0]["input"]["company"] == "Dell"


def test_predict_sama_dengan_model_untuk_baris_dataset(klien):
    """Prediksi lewat API untuk laptop di CSV = prediksi model langsung."""
    c, model_terlatih = klien
    df, _ = bersihkan_data(muat_dataset(ambil_pengaturan().path_dataset))
    F = ekstrak_fitur(df)
    baris = F.iloc[0]  # Apple MacBook Pro 13.3"
    input_ = {
        "company": baris["Company"], "type_name": baris["TypeName"], "inches": baris["Inches"],
        "ram_gb": baris["Ram"], "berat_kg": baris["Weight"], "cpu_kategori": baris["CPU_Kategori"],
        "cpu_ghz": baris["CPU_GHz"], "lebar_px": int(baris["Lebar_px"]), "tinggi_px": int(baris["Tinggi_px"]),
        "touchscreen": bool(baris["Touchscreen"]), "ips": bool(baris["IPS"]), "ssd_gb": baris["SSD_GB"],
        "hdd_gb": baris["HDD_GB"], "flash_gb": baris["Flash_GB"], "gpu_merek": baris["GPU_Merek"], "os": baris["OS"],
    }
    d = c.post("/api/predict", json=input_).json()
    langsung = model_terlatih["ols_normal_equation"].predict_euro(F.iloc[[0]])[0]
    assert d["harga_dataset_eur"] == pytest.approx(langsung, rel=1e-12)


def test_predict_peringatan_ekstrapolasi_dan_kategori_langka(klien):
    c, _ = klien
    r = c.post("/api/predict", json={**INPUT_CONTOH, "ram_gb": 64, "company": "Razer"}).json()
    teks = " ".join(r["peringatan"])
    assert "RAM 64 GB di luar rentang data latih" in teks
    assert "Razer" in teks and "Lainnya" in teks


@pytest.mark.parametrize(
    "ubah, kolom",
    [
        ({"ram_gb": -4}, "ram_gb"),
        ({"inches": 40}, "inches"),
        ({"lebar_px": "lebar"}, "lebar_px"),
        ({"company": "MerekPalsu"}, "company"),
        ({"os": "Windows 11"}, "os"),
    ],
)
def test_predict_input_salah_ditolak(klien, ubah, kolom):
    c, _ = klien
    r = c.post("/api/predict", json={**INPUT_CONTOH, **ubah})
    assert r.status_code == 422
    assert kolom in [k["kolom"] for k in r.json()["detail"]["kesalahan"]]


def test_predict_tanpa_penyimpanan_ditolak(klien):
    c, _ = klien
    r = c.post("/api/predict", json={**INPUT_CONTOH, "ssd_gb": 0, "hdd_gb": 0, "flash_gb": 0})
    assert r.status_code == 422
    assert "penyimpanan" in r.json()["detail"]["kesalahan"][0]["pesan"]


def test_predict_field_wajib_hilang(klien):
    c, _ = klien
    tanpa_ram = {k: v for k, v in INPUT_CONTOH.items() if k != "ram_gb"}
    assert c.post("/api/predict", json=tanpa_ram).status_code == 422


def test_predict_tanpa_model_aktif_503(klien_tanpa_model):
    r = klien_tanpa_model.post("/api/predict", json=INPUT_CONTOH)
    assert r.status_code == 503
    assert "ml.train" in r.json()["detail"]
    assert klien_tanpa_model.get("/api/health").json()["status"] == "bermasalah"


# --------------------------- Model & evaluasi ------------------------------
def test_models_dan_evaluation(klien):
    c, _ = klien
    r = c.get("/api/models").json()
    assert len(r["data"]) == 7
    aktif = [m for m in r["data"] if m["is_active"]]
    assert aktif[0]["id"] == r["model_aktif_id"]

    e = c.get(f"/api/evaluation/{r['model_aktif_id']}").json()
    assert {"metrik", "cv", "residual", "qq_plot", "koefisien", "top_error"} <= set(e["evaluasi"])
    assert len(e["evaluasi"]["top_error"]) == 10
    assert c.get("/api/evaluation/99999").status_code == 404


def test_cors_mengizinkan_frontend(klien):
    c, _ = klien
    r = c.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers.get("access-control-allow-origin") == "http://localhost:3000"
