"""
skema.py — Bentuk data masuk & keluar API (Pydantic).

Pydantic memvalidasi input SEBELUM sampai ke model: tipe data harus benar
dan angka harus masuk akal secara fisik. Input yang salah langsung ditolak
dengan status 422 dan pesan yang menjelaskan kolom mana yang salah.

Batas di sini adalah batas "masuk akal" (mis. RAM 1–128 GB). Batas
pengalaman model (rentang data latih) dicek terpisah dan hanya menghasilkan
peringatan, bukan penolakan.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class InputPrediksi(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "company": "Dell",
                "type_name": "Notebook",
                "inches": 15.6,
                "ram_gb": 8,
                "berat_kg": 2.04,
                "cpu_kategori": "Intel Core i5",
                "cpu_ghz": 2.5,
                "lebar_px": 1920,
                "tinggi_px": 1080,
                "touchscreen": False,
                "ips": False,
                "ssd_gb": 256,
                "hdd_gb": 0,
                "flash_gb": 0,
                "gpu_merek": "Intel",
                "os": "Windows",
            }
        }
    )

    company: str = Field(min_length=1, max_length=40, description="Merek laptop")
    type_name: str = Field(min_length=1, max_length=40, description="Tipe laptop")
    inches: float = Field(ge=7, le=21, description="Ukuran layar (inci)")
    ram_gb: float = Field(ge=1, le=128, description="RAM (GB)")
    berat_kg: float = Field(gt=0.3, le=6, description="Berat (kg)")
    cpu_kategori: str = Field(min_length=1, max_length=40)
    cpu_ghz: float = Field(ge=0.5, le=6, description="Frekuensi dasar CPU (GHz)")
    lebar_px: int = Field(ge=640, le=7680, description="Lebar resolusi layar (piksel)")
    tinggi_px: int = Field(ge=480, le=4320, description="Tinggi resolusi layar (piksel)")
    touchscreen: bool = False
    ips: bool = False
    ssd_gb: float = Field(default=0, ge=0, le=8000)
    hdd_gb: float = Field(default=0, ge=0, le=8000)
    flash_gb: float = Field(default=0, ge=0, le=2000)
    gpu_merek: str = Field(min_length=1, max_length=40)
    os: str = Field(min_length=1, max_length=40)
    # Nama model CPU/GPU yang dipilih di form (mis. "Intel Core i5 7200U").
    # Tidak dipakai model (model memakai cpu_kategori, cpu_ghz, gpu_merek),
    # hanya disimpan di riwayat agar mudah dibaca.
    cpu_model: str | None = Field(default=None, max_length=80)
    gpu_model: str | None = Field(default=None, max_length=80)

    @model_validator(mode="after")
    def _penyimpanan_tidak_kosong(self):
        if self.ssd_gb + self.hdd_gb + self.flash_gb <= 0:
            raise ValueError("Laptop harus memiliki penyimpanan: isi minimal salah satu dari SSD, HDD, atau flash.")
        return self

    def ke_kolom_model(self) -> dict:
        """Nama field API (snake_case) -> nama kolom internal pipeline ML."""
        return {
            "Company": self.company,
            "TypeName": self.type_name,
            "Inches": self.inches,
            "Ram": self.ram_gb,
            "Weight": self.berat_kg,
            "CPU_Kategori": self.cpu_kategori,
            "CPU_GHz": self.cpu_ghz,
            "Lebar_px": self.lebar_px,
            "Tinggi_px": self.tinggi_px,
            "Touchscreen": int(self.touchscreen),
            "IPS": int(self.ips),
            "SSD_GB": self.ssd_gb,
            "HDD_GB": self.hdd_gb,
            "Flash_GB": self.flash_gb,
            "GPU_Merek": self.gpu_merek,
            "OS": self.os,
        }


class ItemKontribusi(BaseModel):
    fitur: str
    label: str
    kontribusi_log: float
    efek_persen: float


class Kontribusi(BaseModel):
    intersep_log: float
    harga_acuan_euro: float
    fitur: list[ItemKontribusi]
    acuan: dict


class ResponsPrediksi(BaseModel):
    prediksi_id: int
    model: dict
    harga_dataset_eur: float = Field(description="Prediksi pada tingkat harga dataset (sekitar 2018)")
    rentang_dataset_eur: list[float]
    harga_sekarang_eur: float = Field(description="Setelah disesuaikan inflasi zona euro")
    rentang_sekarang_eur: list[float]
    harga_idr: float = Field(description="Setelah penyesuaian inflasi & dikonversi ke rupiah")
    rentang_idr: list[float]
    prediksi_log: float
    rentang: dict
    kontribusi: Kontribusi
    peringatan: list[str]
    penyesuaian_harga: dict
    fitur_turunan: dict
