"""
preprocessing.py — Membaca dataset, membuat fitur, dan menyiapkan matriks X.

Alur di file ini:
    CSV mentah
      -> muat_dataset()      : baca CSV, samakan nama kolom, cek kolom wajib
      -> ekstrak_fitur()     : ubah teks menjadi angka (resolusi, memori, CPU, ...)
      -> Preprocessor.fit()  : PELAJARI parameter (rata-rata, std, daftar kategori)
                               HANYA dari data latih
      -> Preprocessor.transform() : ubah tabel fitur menjadi matriks angka X

Format dataset yang dipakai (hasil pemeriksaan file asli, 1275 baris, tanpa
nilai kosong):
    Company, Product, TypeName, Inches, ScreenResolution, CPU_Company,
    CPU_Type, CPU_Frequency (GHz), RAM (GB), Memory, GPU_Company, GPU_Type,
    OpSys, Weight (kg), Price (Euro)
Kolom RAM, berat, dan frekuensi CPU di file ini sudah berupa angka. Fungsi
`ke_angka()` tetap menerima format teks seperti "8GB" / "1.37kg" agar
pipeline juga bisa membaca versi dataset Kaggle yang belum dibersihkan.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Nama kolom
# ---------------------------------------------------------------------------
# Nama kolom di CSV -> nama kolom internal yang dipakai di seluruh kode.
# Versi dataset Kaggle lain memakai nama berbeda (mis. "Ram", "Weight",
# "Price_euros"), jadi keduanya dipetakan ke nama yang sama.
PEMETAAN_KOLOM = {
    "RAM (GB)": "Ram",
    "Ram": "Ram",
    "Weight (kg)": "Weight",
    "Weight": "Weight",
    "CPU_Frequency (GHz)": "CPU_GHz",
    "Price (Euro)": "Price_euros",
    "Price_euros": "Price_euros",
}
KOLOM_WAJIB = [
    "Company", "Product", "TypeName", "Inches", "ScreenResolution",
    "CPU_Company", "CPU_Type", "CPU_GHz", "Ram", "Memory",
    "GPU_Company", "OpSys", "Weight", "Price_euros",
]
KOLOM_TARGET = "Price_euros"

# Fitur numerik yang masuk model (semuanya distandardisasi).
#
# 1) Lebar_px & Tinggi_px tetap dihitung (untuk EDA dan analisis VIF), tetapi
#    TIDAK dimasukkan ke model: VIF keduanya dan PPI sekitar 120 jika dipakai
#    bersamaan (hampir kolinear sempurna). Resolusi sudah terwakili oleh PPI.
#
# 2) RAM dan kapasitas penyimpanan dimasukkan dalam bentuk log2:
#       Ram_log2   = log2(RAM)
#       SSD_log2   = log2(1 + SSD_GB)   (+1 agar laptop tanpa SSD bernilai 0)
#    Alasannya: menambah RAM dari 4 ke 8 GB menaikkan harga jauh lebih besar
#    daripada dari 32 ke 36 GB. Yang berpengaruh adalah "berapa kali lipat",
#    sehingga dengan log2, setiap penggandaan RAM menaikkan harga dengan
#    persentase yang sama. Di 5-Fold CV (data latih), perubahan ini
#    menurunkan RMSE regresi linear dari 389 ke 360 euro. Bentuk linear
#    membuat laptop RAM 64 GB ditebak jauh terlalu mahal (ekstrapolasi).
FITUR_NUMERIK = ["Inches", "Ram_log2", "Weight", "CPU_GHz", "PPI", "SSD_log2", "HDD_log2", "Flash_log2"]
FITUR_NUMERIK_SEMUA = ["Inches", "Ram_log2", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px",
                       "PPI", "SSD_log2", "HDD_log2", "Flash_log2"]
# Kolom mentah (satuan asli) yang disimpan untuk EDA dan form input
FITUR_SATUAN_ASLI = ["Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px",
                     "PPI", "SSD_GB", "HDD_GB", "Flash_GB"]
# Fitur biner (0/1) TIDAK distandardisasi agar koefisiennya mudah dibaca:
# "laptop dengan layar sentuh lebih mahal sekian persen".
FITUR_BINER = ["Touchscreen", "IPS"]
# Fitur kategori -> one-hot encoding
FITUR_KATEGORI = ["Company", "TypeName", "CPU_Kategori", "GPU_Merek", "OS"]
# Fitur numerik utama yang diberi suku kuadrat & interaksi (model polinomial)
FITUR_POLINOMIAL = ["Ram_log2", "PPI", "SSD_log2", "Weight", "CPU_GHz"]

KATEGORI_LAINNYA = "Lainnya"


# ===========================================================================
# 1. Membaca dataset
# ===========================================================================
def muat_dataset(path: str | Path) -> pd.DataFrame:
    """Membaca CSV dan menyeragamkan nama kolom.

    Melempar error yang jelas jika ada kolom wajib yang tidak ditemukan,
    supaya tidak terjadi kesalahan diam-diam karena format file berbeda.
    """
    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in PEMETAAN_KOLOM.items() if k in df.columns})
    hilang = [k for k in KOLOM_WAJIB if k not in df.columns]
    if hilang:
        raise ValueError(f"Kolom wajib tidak ditemukan di dataset: {hilang}")
    return df


# ===========================================================================
# 2. Fungsi parsing per kolom (fungsi murni -> mudah diuji satu per satu)
# ===========================================================================
def ke_angka(nilai, satuan: str = "") -> float:
    """Mengubah nilai seperti 8, "8", "8GB", atau "1.37kg" menjadi float."""
    if isinstance(nilai, (int, float, np.integer, np.floating)):
        return float(nilai)
    teks = str(nilai).strip().lower().replace(satuan.lower(), "").strip()
    return float(teks)


def parse_resolusi(teks: str) -> dict:
    """Memecah kolom ScreenResolution.

    Contoh: "IPS Panel Full HD / Touchscreen 1920x1080"
        -> {"Lebar_px": 1920, "Tinggi_px": 1080, "Touchscreen": 1, "IPS": 1}

    Label seperti "Full HD", "4K Ultra HD", atau "Retina Display" diabaikan
    karena informasinya sudah ada di angka resolusi. (Di dataset bahkan ada
    label yang tidak konsisten, mis. "IPS Panel Full HD 1366x768".)
    """
    cocok = re.search(r"(\d+)\s*x\s*(\d+)", teks)
    if not cocok:
        raise ValueError(f"Format resolusi tidak dikenali: {teks!r}")
    return {
        "Lebar_px": int(cocok.group(1)),
        "Tinggi_px": int(cocok.group(2)),
        "Touchscreen": int("touchscreen" in teks.lower()),
        "IPS": int("ips" in teks.lower()),
    }


def hitung_ppi(lebar_px: float, tinggi_px: float, inci: float) -> float:
    """Pixels Per Inch = panjang diagonal dalam piksel / panjang diagonal dalam inci.

    Diagonal piksel dihitung dengan teorema Pythagoras: sqrt(lebar^2 + tinggi^2).
    PPI menggabungkan resolusi dan ukuran layar menjadi satu ukuran
    "ketajaman layar".
    """
    return float(np.sqrt(lebar_px ** 2 + tinggi_px ** 2) / inci)


def kategori_cpu(perusahaan: str, tipe: str) -> str:
    """Mengelompokkan prosesor.

    Kategori: Intel Core i7 / i5 / i3, Intel Lainnya, AMD, Lainnya.

    "Intel Lainnya" berisi Celeron, Pentium, Atom, Core M, dan Xeon
    (142 baris). Kelompok ini dipisah dari "Lainnya" (prosesor non-Intel
    non-AMD, hanya 1 baris: Samsung Cortex ARM) agar 142 laptop Intel kelas
    bawah tidak tercampur dengan kategori yang hampir kosong.
    """
    perusahaan = perusahaan.strip()
    tipe = tipe.strip()
    if perusahaan == "AMD":
        return "AMD"
    if perusahaan != "Intel":
        return KATEGORI_LAINNYA
    for seri in ("i7", "i5", "i3"):
        if re.match(rf"Core {seri}\b", tipe):
            return f"Intel Core {seri}"
    return "Intel Lainnya"


def parse_memori(teks: str) -> dict:
    """Menghitung kapasitas penyimpanan per jenis, dalam GB.

    Contoh: "128GB SSD +  1TB HDD" -> SSD 128, HDD 1000, Flash 0

    Aturan:
    - Teks dipisah dengan tanda "+" (di dataset ada dua spasi: "SSD +  1TB").
    - 1 TB = 1000 GB (satuan desimal yang dipakai produsen penyimpanan).
    - "Flash Storage" (71 baris) disimpan di kolom sendiri, Flash_GB. Umumnya
      berupa eMMC kapasitas kecil (16-64 GB) pada laptop murah, sehingga
      tidak adil disamakan dengan SSD.
    - "Hybrid" (SSHD, 10 baris) dihitung sebagai HDD, karena pada dasarnya
      adalah hard disk piringan dengan sedikit cache flash.
    - Dua drive sejenis dijumlahkan ("256GB SSD + 256GB SSD" -> SSD 512).
    """
    hasil = {"SSD_GB": 0.0, "HDD_GB": 0.0, "Flash_GB": 0.0}
    for bagian in teks.split("+"):
        bagian = bagian.strip()
        cocok = re.match(r"([\d.]+)\s*(GB|TB)\s+(.+)", bagian, flags=re.IGNORECASE)
        if not cocok:
            raise ValueError(f"Format memori tidak dikenali: {bagian!r}")
        kapasitas = float(cocok.group(1)) * (1000 if cocok.group(2).upper() == "TB" else 1)
        jenis = cocok.group(3).strip().lower()
        if jenis == "ssd":
            hasil["SSD_GB"] += kapasitas
        elif jenis in ("hdd", "hybrid"):
            hasil["HDD_GB"] += kapasitas
        elif jenis == "flash storage":
            hasil["Flash_GB"] += kapasitas
        else:
            raise ValueError(f"Jenis penyimpanan tidak dikenali: {jenis!r}")
    return hasil


def merek_gpu(perusahaan: str) -> str:
    """Intel / Nvidia / AMD. Selain itu (1 baris GPU ARM Mali) -> Lainnya."""
    perusahaan = perusahaan.strip()
    return perusahaan if perusahaan in ("Intel", "Nvidia", "AMD") else KATEGORI_LAINNYA


def sederhanakan_os(os_asli: str) -> str:
    """Menyederhanakan 9 nilai OpSys menjadi 5 kelompok.

    Windows 10 / Windows 10 S / Windows 7 -> Windows
    macOS / Mac OS X                      -> macOS
    No OS                                 -> Tanpa OS
    Linux                                 -> Linux
    Chrome OS / Android                   -> Chrome OS/Android
        (Android hanya 2 baris; digabung dengan Chrome OS karena sama-sama
         OS Google untuk perangkat kelas bawah)
    """
    teks = os_asli.strip().lower()
    if teks.startswith("windows"):
        return "Windows"
    if teks in ("macos", "mac os x"):
        return "macOS"
    if teks == "no os":
        return "Tanpa OS"
    if teks == "linux":
        return "Linux"
    if teks in ("chrome os", "android"):
        return "Chrome OS/Android"
    return KATEGORI_LAINNYA


def bersihkan_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Membuang baris yang di luar cakupan model, dengan alasan tercatat.

    Saat ini hanya satu aturan: laptop dengan prosesor selain Intel/AMD.
    Di dataset hanya ada 1 baris seperti ini (Samsung Chromebook Plus, CPU
    ARM Cortex, GPU ARM Mali). Satu data tidak cukup untuk mempelajari efek
    kategori, dan kolom one-hot CPU=Lainnya serta GPU=Lainnya akan identik
    (kolinearitas sempurna). Model ini dinyatakan berlaku untuk laptop
    berprosesor Intel/AMD.
    """
    luar_cakupan = ~df["CPU_Company"].astype(str).str.strip().isin(["Intel", "AMD"])
    dibuang = [
        {
            "indeks": int(i),
            "company": df.at[i, "Company"],
            "product": df.at[i, "Product"],
            "alasan": f"CPU {df.at[i, 'CPU_Company']} {df.at[i, 'CPU_Type']} (bukan Intel/AMD)",
        }
        for i in df.index[luar_cakupan]
    ]
    return df.loc[~luar_cakupan].reset_index(drop=True), dibuang


# ===========================================================================
# 3. Feature engineering untuk seluruh tabel
# ===========================================================================
def ekstrak_fitur(df: pd.DataFrame) -> pd.DataFrame:
    """Membuat tabel fitur dari data mentah (baris per baris).

    Langkah ini TIDAK mempelajari apa pun dari data (setiap baris diproses
    sendiri-sendiri), sehingga aman dilakukan sebelum split latih/uji tanpa
    menyebabkan data leakage. Yang berpotensi bocor adalah langkah yang
    menghitung statistik dari banyak baris (rata-rata, std, frekuensi
    kategori) — itu dikerjakan di Preprocessor.fit() khusus data latih.
    """
    hasil = pd.DataFrame(index=df.index)

    # --- kolom yang dipakai apa adanya ---
    hasil["Company"] = df["Company"].astype(str).str.strip()
    hasil["TypeName"] = df["TypeName"].astype(str).str.strip()
    hasil["Inches"] = df["Inches"].map(ke_angka)

    # --- angka dari kolom RAM, berat, frekuensi CPU ---
    hasil["Ram"] = df["Ram"].map(lambda v: ke_angka(v, "gb"))
    hasil["Weight"] = df["Weight"].map(lambda v: ke_angka(v, "kg"))
    hasil["CPU_GHz"] = df["CPU_GHz"].map(lambda v: ke_angka(v, "ghz"))

    # --- layar ---
    layar = pd.DataFrame(df["ScreenResolution"].map(parse_resolusi).tolist(), index=df.index)
    hasil = hasil.join(layar)

    # --- penyimpanan ---
    memori = pd.DataFrame(df["Memory"].map(parse_memori).tolist(), index=df.index)
    hasil = hasil.join(memori)

    # --- kategori hasil penyederhanaan ---
    hasil["CPU_Kategori"] = [kategori_cpu(p, t) for p, t in zip(df["CPU_Company"], df["CPU_Type"])]
    hasil["GPU_Merek"] = df["GPU_Company"].map(merek_gpu)
    hasil["OS"] = df["OpSys"].map(sederhanakan_os)
    return tambah_fitur_turunan(hasil)


def tambah_fitur_turunan(fitur: pd.DataFrame) -> pd.DataFrame:
    """Menghitung PPI dan fitur log2 dari kolom satuan asli.

    Dipakai oleh ekstrak_fitur() (data CSV) DAN fitur_dari_input() (form
    website), sehingga rumusnya dijamin sama persis di kedua tempat.
    """
    fitur = fitur.copy()
    fitur["PPI"] = [
        hitung_ppi(l, t, i) for l, t, i in zip(fitur["Lebar_px"], fitur["Tinggi_px"], fitur["Inches"])
    ]
    fitur["Ram_log2"] = np.log2(fitur["Ram"].astype(float))
    for jenis in ("SSD", "HDD", "Flash"):
        fitur[f"{jenis}_log2"] = np.log2(1.0 + fitur[f"{jenis}_GB"].astype(float))
    return fitur


def target_log(df: pd.DataFrame) -> np.ndarray:
    """Target model: y = ln(harga dalam euro).

    Kenapa memakai logaritma harga, bukan harga langsung?
    1. Distribusi harga miring ke kanan (skewness 1,51 di dataset ini):
       banyak laptop murah, sedikit yang sangat mahal. Setelah di-log
       skewness menjadi -0,18 (hampir simetris), lebih cocok dengan asumsi
       regresi linear bahwa error berdistribusi normal.
    2. Pengaruh spesifikasi terhadap harga bersifat persentase (perkalian),
       bukan penambahan tetap. Menambah RAM menaikkan harga laptop 300 euro
       dan 3000 euro dengan persentase yang mirip, bukan dengan nominal yang
       sama. Di skala log, efek perkalian menjadi penjumlahan -> linear.
    3. Variansi error lebih seragam (homoskedastis): tanpa log, error
       laptop mahal jauh lebih besar daripada laptop murah.
    4. Hasil prediksi exp(y_hat) pasti positif — tidak mungkin harga negatif.
    """
    return np.log(df[KOLOM_TARGET].to_numpy(dtype=float))


KOLOM_INPUT = ["Company", "TypeName", "Inches", "Ram", "Weight", "CPU_GHz", "Lebar_px", "Tinggi_px",
               "Touchscreen", "IPS", "SSD_GB", "HDD_GB", "Flash_GB", "CPU_Kategori", "GPU_Merek", "OS"]


def fitur_dari_input(data: dict) -> pd.DataFrame:
    """Membuat tabel fitur 1 baris dari input form (dipakai API /predict).

    Input sudah berupa spesifikasi terstruktur (bukan teks mentah CSV),
    jadi cukup hitung fitur turunan (PPI, log2) dengan fungsi yang sama.
    """
    hilang = [k for k in KOLOM_INPUT if k not in data]
    if hilang:
        raise ValueError(f"Input kurang lengkap: {hilang}")
    return tambah_fitur_turunan(pd.DataFrame([{k: data[k] for k in KOLOM_INPUT}]))


# ===========================================================================
# 4. Preprocessor: one-hot encoding + standardisasi (fit hanya di data latih)
# ===========================================================================
class Preprocessor:
    """Mengubah tabel fitur menjadi matriks angka X yang siap dipakai model.

    fit(data_latih) mempelajari:
      - daftar kategori tiap fitur kategori (kategori yang muncul kurang dari
        `min_frekuensi` kali di data latih digabung menjadi "Lainnya"),
      - kategori acuan (yang paling sering muncul) yang kolom one-hot-nya
        dibuang (drop-first),
      - rata-rata dan standar deviasi tiap fitur numerik.

    transform(data_apa_pun) memakai parameter hasil fit tersebut. Data uji
    dan data baru TIDAK pernah dipakai untuk menghitung rata-rata/std, karena
    jika dipakai, model secara tidak langsung "mengintip" data uji (data
    leakage) dan skor evaluasinya menjadi terlalu optimis.

    Kenapa satu kolom one-hot dibuang (drop-first)?
    Jika semua kategori diberi kolom, jumlah kolom one-hot selalu = 1 untuk
    setiap baris, persis sama dengan kolom bias (angka 1). Akibatnya ada
    kolinearitas sempurna (dummy variable trap) dan koefisien tidak unik.
    Dengan membuang kategori acuan, koefisien kategori lain dibaca sebagai
    "selisih harga dibanding kategori acuan".

    Pengaman kolinearitas sempurna lain (dicek otomatis di data latih):
    - Kolom one-hot yang isinya identik dengan kolom sebelumnya dibuang dan
      dicatat di `kolom_duplikat_`. Di dataset ini: OS=macOS identik dengan
      Company=Apple (21 laptop Apple semuanya macOS dan sebaliknya), sehingga
      efek "Apple" di model = efek merek Apple + macOS sekaligus.
    - Jika kelompok "Lainnya" hasil penggabungan masih lebih sedikit dari
      `min_frekuensi`, kelompok itu tidak diberi kolom sendiri (baris-baris
      tersebut diperlakukan sama dengan kategori acuan), karena koefisien
      yang dipelajari dari 1-4 data hanya akan menghafal data itu.
    """

    def __init__(
        self,
        fitur_numerik: list[str] | None = None,
        fitur_biner: list[str] | None = None,
        fitur_kategori: list[str] | None = None,
        fitur_polinomial: list[str] | None = None,
        min_frekuensi: int = 5,
        buang_kolom_duplikat: bool = True,
    ) -> None:
        # buang_kolom_duplikat=False hanya dipakai untuk analisis
        # multikolinearitas ("bagaimana jika kolom identik tidak dibuang?")
        self.buang_kolom_duplikat = buang_kolom_duplikat
        self.fitur_numerik = list(fitur_numerik if fitur_numerik is not None else FITUR_NUMERIK)
        self.fitur_biner = list(fitur_biner if fitur_biner is not None else FITUR_BINER)
        self.fitur_kategori = list(fitur_kategori if fitur_kategori is not None else FITUR_KATEGORI)
        # None = tanpa suku polinomial (regresi linear biasa)
        self.fitur_polinomial = list(fitur_polinomial) if fitur_polinomial else []
        self.min_frekuensi = min_frekuensi

    # ------------------------------------------------------------------
    def _rapikan_kategori(self, kolom: str, nilai: pd.Series) -> pd.Series:
        """Nilai yang tidak dikenal / langka diganti menjadi 'Lainnya'."""
        dikenal = set(self.kategori_[kolom])
        return nilai.astype(str).where(nilai.astype(str).isin(dikenal), KATEGORI_LAINNYA)

    def fit(self, df_fitur: pd.DataFrame) -> "Preprocessor":
        # --- kategori ---
        self.kategori_: dict[str, list[str]] = {}
        self.kategori_acuan_: dict[str, str] = {}
        for kolom in self.fitur_kategori:
            frekuensi = df_fitur[kolom].astype(str).value_counts()
            cukup = frekuensi[frekuensi >= self.min_frekuensi].index.tolist()
            # jumlah baris yang akan masuk kelompok "Lainnya"
            n_lainnya = int(frekuensi[~frekuensi.index.isin(cukup)].sum())
            if n_lainnya >= self.min_frekuensi and KATEGORI_LAINNYA not in cukup:
                cukup.append(KATEGORI_LAINNYA)
            self.kategori_[kolom] = cukup
            # acuan = kategori terbanyak (sudah urutan pertama value_counts)
            self.kategori_acuan_[kolom] = cukup[0]

        # --- rentang nilai data latih dalam satuan asli (untuk peringatan
        #     ekstrapolasi di API: input di luar rentang ini tidak pernah
        #     "dilihat" model, sehingga prediksinya kurang bisa dipercaya) ---
        self.rentang_latih_ = {
            k: (float(df_fitur[k].min()), float(df_fitur[k].max()))
            for k in FITUR_SATUAN_ASLI
            if k in df_fitur.columns
        }

        # --- numerik: rata-rata & std dari data latih ---
        numerik = df_fitur[self.fitur_numerik].to_numpy(dtype=float)
        self.rata_rata_ = numerik.mean(axis=0)
        self.std_ = numerik.std(axis=0)
        self.std_[self.std_ == 0] = 1.0   # cegah pembagian dengan nol

        # --- one-hot: cari kolom yang identik dengan kolom sebelumnya ---
        self.kolom_onehot_: list[tuple[str, str]] = []   # (fitur, kategori)
        self.kolom_duplikat_: list[dict] = []
        sudah_ada: dict[bytes, str] = {}
        for kolom in self.fitur_kategori:
            nilai = self._rapikan_kategori(kolom, df_fitur[kolom])
            for k in self.kategori_[kolom]:
                if k == self.kategori_acuan_[kolom]:
                    continue
                isi = (nilai == k).to_numpy(dtype=np.uint8).tobytes()
                nama_kolom = f"{kolom}={k}"
                if isi in sudah_ada and self.buang_kolom_duplikat:
                    self.kolom_duplikat_.append({"dibuang": nama_kolom, "identik_dengan": sudah_ada[isi]})
                    continue
                sudah_ada[isi] = nama_kolom
                self.kolom_onehot_.append((kolom, k))

        # --- susunan nama kolom keluaran ---
        nama = list(self.fitur_numerik) + list(self.fitur_biner)
        nama += [f"{kolom}={k}" for kolom, k in self.kolom_onehot_]
        nama += [n for n, _, _ in self._pasangan_polinomial()]
        self.nama_fitur_ = nama
        return self

    def _pasangan_polinomial(self) -> list[tuple[str, int, int]]:
        """Daftar suku derajat 2: kuadrat (a*a) dan interaksi (a*b)."""
        idx = [self.fitur_numerik.index(f) for f in self.fitur_polinomial]
        pasangan = []
        for i_pos, i in enumerate(idx):
            for j in idx[i_pos:]:
                a, b = self.fitur_numerik[i], self.fitur_numerik[j]
                nama = f"{a}^2" if i == j else f"{a}*{b}"
                pasangan.append((nama, i, j))
        return pasangan

    def transform(self, df_fitur: pd.DataFrame) -> np.ndarray:
        # numerik distandardisasi: z = (x - rata2_latih) / std_latih
        numerik = (df_fitur[self.fitur_numerik].to_numpy(dtype=float) - self.rata_rata_) / self.std_
        bagian = [numerik, df_fitur[self.fitur_biner].to_numpy(dtype=float)]

        nilai_rapi = {kolom: self._rapikan_kategori(kolom, df_fitur[kolom]) for kolom in self.fitur_kategori}
        for kolom, k in self.kolom_onehot_:
            bagian.append((nilai_rapi[kolom] == k).to_numpy(dtype=float).reshape(-1, 1))

        # suku polinomial dihitung dari fitur yang SUDAH distandardisasi
        for _, i, j in self._pasangan_polinomial():
            bagian.append((numerik[:, i] * numerik[:, j]).reshape(-1, 1))

        return np.hstack(bagian)

    def fit_transform(self, df_fitur: pd.DataFrame) -> np.ndarray:
        return self.fit(df_fitur).transform(df_fitur)

    def fitur_asal(self, nama_kolom: str) -> str:
        """Nama fitur asli dari suatu kolom X (untuk mengelompokkan kontribusi).

        "Company=Apple" -> "Company", "Ram^2" / "Ram*PPI" -> "Polinomial", "Ram" -> "Ram"
        """
        if "=" in nama_kolom:
            return nama_kolom.split("=", 1)[0]
        if "^" in nama_kolom or "*" in nama_kolom:
            return "Polinomial"
        return nama_kolom
