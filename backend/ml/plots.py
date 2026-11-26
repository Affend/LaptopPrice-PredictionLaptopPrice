"""
plots.py — Ekspor grafik PNG ke backend/outputs/ untuk laporan & slide.

Website TIDAK memakai gambar ini (grafik di website digambar ulang secara
interaktif dari data API). Semua gambar dibuat dari hasil yang sama yang
disimpan ke database, jadi angkanya pasti konsisten.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # tanpa jendela GUI (aman dijalankan di server/Docker)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WARNA_UTAMA = "#2563eb"
WARNA_KEDUA = "#f97316"
WARNA_ABU = "#6b7280"


def _simpan(fig, folder: Path, nama: str) -> str:
    path = folder / nama
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def ekspor_semua(hasil: dict, df_bersih: pd.DataFrame, F: pd.DataFrame, folder: Path) -> list[str]:
    """Membuat semua gambar dari dict hasil train.jalankan_pipeline()."""
    folder.mkdir(parents=True, exist_ok=True)
    terbaik = hasil["model_terbaik"]
    m = hasil["model"][terbaik]
    label = hasil["model"][terbaik]["label"]
    file = []

    # 1. Distribusi harga: asli vs log
    harga = df_bersih["Price_euros"].to_numpy(float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(harga, bins=40, color=WARNA_UTAMA, alpha=0.85)
    ax[0].set(title=f"Harga asli (skewness {pd.Series(harga).skew():.2f})", xlabel="Harga (euro)", ylabel="Jumlah laptop")
    ax[1].hist(np.log(harga), bins=40, color=WARNA_KEDUA, alpha=0.85)
    ax[1].set(title=f"ln(harga) (skewness {pd.Series(np.log(harga)).skew():.2f})", xlabel="ln(harga)")
    file.append(_simpan(fig, folder, "01_distribusi_harga.png"))

    # 2. Korelasi fitur numerik dengan ln(harga)
    kolom = ["Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px", "PPI", "SSD_GB", "HDD_GB", "Flash_GB"]
    korelasi = pd.concat([F[kolom], pd.Series(np.log(harga), name="ln_harga", index=F.index)], axis=1).corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    gambar = ax.imshow(korelasi.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(korelasi)), korelasi.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(korelasi)), korelasi.columns)
    for i in range(len(korelasi)):
        for j in range(len(korelasi)):
            ax.text(j, i, f"{korelasi.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(gambar, ax=ax, shrink=0.8)
    ax.set_title("Korelasi Pearson fitur numerik")
    file.append(_simpan(fig, folder, "02_korelasi.png"))

    # 3. Perbandingan model: RMSE CV (± std) dan RMSE uji
    nama_model = list(hasil["model"].keys())
    cv_rata = [hasil["model"][n]["cv"]["ringkasan"]["rmse"]["rata_rata"] for n in nama_model]
    cv_std = [hasil["model"][n]["cv"]["ringkasan"]["rmse"]["std"] for n in nama_model]
    uji = [hasil["model"][n]["metrik"]["uji"]["rmse"] for n in nama_model]
    posisi = np.arange(len(nama_model))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(posisi - 0.2, cv_rata, 0.4, yerr=cv_std, capsize=3, label="RMSE 5-Fold CV (latih)", color=WARNA_UTAMA)
    ax.bar(posisi + 0.2, uji, 0.4, label="RMSE data uji", color=WARNA_KEDUA)
    ax.set_xticks(posisi, nama_model, rotation=20, ha="right")
    ax.set(ylabel="RMSE (euro)", title="Perbandingan model terhadap baseline")
    ax.legend()
    file.append(_simpan(fig, folder, "03_perbandingan_model.png"))

    # 4. Prediksi vs aktual (data uji, model terbaik)
    titik = pd.DataFrame(m["residual"]["titik"])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(titik["aktual_euro"], titik["prediksi_euro"], s=12, alpha=0.6, color=WARNA_UTAMA)
    batas = [0, max(titik["aktual_euro"].max(), titik["prediksi_euro"].max()) * 1.05]
    ax.plot(batas, batas, "--", color=WARNA_ABU, label="prediksi = aktual")
    ax.set(xlabel="Harga aktual (euro)", ylabel="Harga prediksi (euro)", title=f"Prediksi vs aktual — {label}")
    ax.legend()
    file.append(_simpan(fig, folder, "04_prediksi_vs_aktual.png"))

    # 5. Residual vs prediksi (skala log)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(np.log(titik["prediksi_euro"]), titik["residual_log"], s=12, alpha=0.6, color=WARNA_UTAMA)
    ax.axhline(0, color=WARNA_ABU, linestyle="--")
    ax.set(xlabel="Prediksi ln(harga)", ylabel="Residual ln(harga)", title="Residual vs prediksi (data uji)")
    file.append(_simpan(fig, folder, "05_residual_vs_prediksi.png"))

    # 6. Histogram residual + kurva normal
    hist = pd.DataFrame(m["residual"]["histogram"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(hist["tengah"], hist["frekuensi"], width=(hist["bin_kanan"] - hist["bin_kiri"]) * 0.95, color=WARNA_UTAMA, alpha=0.8, label="Residual")
    ax.plot(hist["tengah"], hist["frekuensi_normal"], color=WARNA_KEDUA, linewidth=2, label="Kurva normal")
    ax.set(xlabel="Residual ln(harga)", ylabel="Frekuensi", title="Histogram residual (data uji)")
    ax.legend()
    file.append(_simpan(fig, folder, "06_histogram_residual.png"))

    # 7. Q-Q plot
    qq = pd.DataFrame(m["qq_plot"]["titik"])
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(qq["teoretis"], qq["sampel"], s=10, color=WARNA_UTAMA)
    batas = [qq["teoretis"].min(), qq["teoretis"].max()]
    ax.plot(batas, batas, "--", color=WARNA_ABU)
    ax.set(xlabel="Kuantil teoretis normal", ylabel="Kuantil residual terstandar", title="Q-Q plot residual")
    file.append(_simpan(fig, folder, "07_qq_plot.png"))

    # 8. Learning curve
    lc = pd.DataFrame(m["learning_curve"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lc["jumlah_data_latih"], lc["rmse_latih"], "o-", color=WARNA_UTAMA, label="Error latih")
    ax.fill_between(lc["jumlah_data_latih"], lc["rmse_validasi"] - lc["rmse_validasi_std"], lc["rmse_validasi"] + lc["rmse_validasi_std"], color=WARNA_KEDUA, alpha=0.15)
    ax.plot(lc["jumlah_data_latih"], lc["rmse_validasi"], "o-", color=WARNA_KEDUA, label="Error validasi")
    ax.set(xlabel="Jumlah data latih", ylabel="RMSE (euro)", title=f"Learning curve — {label}")
    ax.legend()
    file.append(_simpan(fig, folder, "08_learning_curve.png"))

    # 9. Kurva loss Gradient Descent (beberapa learning rate)
    gd = hasil["model"]["ols_gradient_descent"]["loss_history"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kurva in gd["perbandingan_learning_rate"]:
        ax.plot(kurva["iterasi"], kurva["loss"], label=f"lr = {kurva['learning_rate']}")
    ax.axhline(gd["loss_normal_equation"], color=WARNA_ABU, linestyle="--", label="MSE Normal Equation")
    ax.set_yscale("log")
    ax.set(xlabel="Iterasi", ylabel="MSE (skala log)", title="Kurva loss Gradient Descent")
    ax.legend()
    file.append(_simpan(fig, folder, "09_kurva_loss_gd.png"))

    # 10. Error per merek (MAPE)
    merek = pd.DataFrame(m["error_per_merek"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(merek["grup"], merek["mape"], color=WARNA_UTAMA)
    for i, n in enumerate(merek["n"]):
        ax.text(i, merek["mape"].iloc[i], f"n={int(n)}", ha="center", va="bottom", fontsize=7)
    ax.set(ylabel="MAPE (%)", title="Error per merek (data uji)")
    ax.tick_params(axis="x", rotation=45)
    file.append(_simpan(fig, folder, "10_error_per_merek.png"))

    # 11. Koefisien model regresi linear (efek % terhadap harga)
    koef = pd.DataFrame(hasil["model"]["ols_normal_equation"]["koefisien"])
    koef = koef.reindex(koef["efek_persen"].abs().sort_values(ascending=False).index).head(15)
    koef = koef.sort_values("efek_persen")
    fig, ax = plt.subplots(figsize=(8, 6))
    warna = [WARNA_KEDUA if v < 0 else WARNA_UTAMA for v in koef["efek_persen"]]
    ax.barh(koef["fitur"], koef["efek_persen"], color=warna)
    ax.axvline(0, color=WARNA_ABU)
    ax.set(xlabel="Efek terhadap harga (%)", title="15 koefisien terbesar — Regresi Linear (Normal Equation)")
    file.append(_simpan(fig, folder, "11_koefisien.png"))

    # 12. Pemilihan alpha Ridge
    alpha = pd.DataFrame(hasil["pemilihan_alpha"]["kandidat"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(alpha["alpha"], alpha["rmse_cv"], yerr=alpha["rmse_cv_std"], fmt="o-", capsize=3, color=WARNA_UTAMA)
    ax.axvline(hasil["pemilihan_alpha"]["alpha_terbaik"], color=WARNA_KEDUA, linestyle="--", label="alpha terpilih")
    ax.set_xscale("log")
    ax.set(xlabel="alpha (skala log)", ylabel="RMSE 5-Fold CV (euro)", title="Pemilihan alpha Ridge")
    ax.legend()
    file.append(_simpan(fig, folder, "12_pemilihan_alpha.png"))

    return file
