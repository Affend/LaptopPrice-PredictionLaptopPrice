/** tipe.ts — Bentuk data yang dikirim backend (lihat backend/app/rute.py). */

export interface InfoModel {
  id: number;
  nomor_versi: number;
  nama_model: string;
  label: string;
  dibuat_pada: string | null;
  rmse_log_cv: number;
}

export interface Health {
  status: "ok" | "bermasalah";
  database: string;
  model_aktif: InfoModel | null;
  pesan_model: string | null;
  jumlah_data: number;
}

export interface Metrik {
  mae: number;
  rmse: number;
  mape: number;
  r2: number;
  r2_log: number;
  rmse_log: number;
}

export type RingkasanCV = Record<string, { rata_rata: number; std: number }>;

export interface Interval {
  z: number;
  rmse_log_cv: number;
  target_cakupan_persen: number;
  cakupan_uji_persen: number;
  jumlah_di_luar: number;
  jumlah_uji: number;
  batas_bawah_relatif_persen: number;
  batas_atas_relatif_persen: number;
}

export interface RingkasModel {
  id: number;
  nomor_versi: number;
  nama_model: string;
  label: string;
  hyperparameter: Record<string, unknown>;
  dibuat_pada: string | null;
  is_active: boolean;
  metrik_uji: Metrik;
  metrik_latih: Metrik;
  perbaikan_vs_baseline: Record<"mae" | "rmse" | "mape", { uji_persen: number; cv_persen: number }>;
  jumlah_fitur: number;
  keterangan: string;
  cv: RingkasanCV;
  interval: Interval | null;
}

export interface DaftarModel {
  model_aktif_id: number | null;
  data: RingkasModel[];
}

export interface Fold extends Metrik {
  fold: number;
  rmse_latih: number;
  r2_latih: number;
}

export interface TitikResidual {
  aktual_euro: number;
  prediksi_euro: number;
  residual_euro: number;
  residual_log: number;
  company: string;
  product: string;
  typename: string;
}

export interface BinHistogram {
  bin_kiri: number;
  bin_kanan: number;
  tengah: number;
  frekuensi: number;
  frekuensi_normal?: number;
}

export interface ErrorGrup {
  grup: string;
  n: number;
  rata_rata_aktual: number;
  rata_rata_prediksi: number;
  mae: number;
  mape: number;
  bias_rata_rata: number;
}

export interface Koefisien {
  fitur: string;
  jenis: "numerik" | "biner" | "kategori" | "polinomial";
  koefisien: number;
  efek_persen: number;
  efek_per_penggandaan_persen: number | null;
  keterangan: string;
}

export interface KurvaLoss {
  iterasi: number[];
  loss: number[];
}

export interface LossHistory extends KurvaLoss {
  learning_rate: number;
  n_iterasi_maks: number;
  n_iterasi_dijalankan: number;
  loss_awal: number;
  loss_akhir: number;
  loss_normal_equation: number;
  eigen_hessian_maks: number;
  eigen_hessian_min: number;
  learning_rate_maks_stabil: number;
  perbandingan_learning_rate: (KurvaLoss & { learning_rate: number; status: string; loss_akhir: number | null })[];
}

export interface Evaluasi {
  model: RingkasModel;
  evaluasi: {
    metrik: { latih: Metrik; uji: Metrik; interval: Interval };
    verifikasi: { metrik: string; manual: number; sklearn: number; selisih_absolut: number; sama: boolean }[];
    cv: { k: number; per_fold: Fold[]; ringkasan: RingkasanCV };
    residual: {
      titik: TitikResidual[];
      histogram: BinHistogram[];
      statistik: {
        n: number;
        rata_rata_residual_log: number;
        std_residual_log: number;
        skewness: number;
        kurtosis_excess: number;
        shapiro_w: number;
        shapiro_p: number;
        korelasi_abs_residual_vs_prediksi: number;
      };
    };
    qq_plot: { titik: { teoretis: number; sampel: number }[] };
    learning_curve?: {
      proporsi: number;
      jumlah_data_latih: number;
      rmse_latih: number;
      rmse_latih_std: number;
      rmse_validasi: number;
      rmse_validasi_std: number;
    }[];
    loss_history?: LossHistory;
    error_per_merek: ErrorGrup[];
    error_per_tipe: ErrorGrup[];
    top_error: {
      peringkat: number;
      company: string;
      product: string;
      typename: string;
      cpu_type: string;
      ram: string;
      memory: string;
      gpu_type: string;
      aktual_euro: number;
      prediksi_euro: number;
      error_euro: number;
      error_persen: number;
      arah: string;
    }[];
    koefisien?: Koefisien[];
  };
  perbandingan_run: RingkasModel[];
  ringkasan_run: RingkasanRun;
}

export interface RingkasanRun {
  nomor_versi: number;
  dibuat_pada: string | null;
  durasi_detik: number;
  dataset: {
    jumlah_baris_mentah: number;
    baris_dibuang: { indeks: number; company: string; product: string; alasan: string }[];
    jumlah_baris_dipakai: number;
    jumlah_latih: number;
    jumlah_uji: number;
  };
  pemilihan_alpha: {
    kandidat: { alpha: number; rmse_cv: number; rmse_cv_std: number; mape_cv: number; r2_cv: number }[];
    alpha_terbaik: number;
  };
  verifikasi_koefisien: {
    perbandingan: string;
    model_a: string;
    model_b: string;
    selisih_maks_koefisien: number;
    selisih_intersep: number;
    selisih_maks_prediksi_log: number;
    jumlah_koefisien: number;
  }[];
  multikolinearitas: {
    vif_semua_fitur_numerik: { fitur: string; vif: number; r2_terhadap_fitur_lain: number }[];
    vif_fitur_dipakai: { fitur: string; vif: number; r2_terhadap_fitur_lain: number }[];
    skenario_condition_number: {
      skenario: string;
      jumlah_kolom: number;
      rank: number;
      singular: boolean;
      condition_number: number | null;
    }[];
    jalur_koefisien_ridge_fitur_layar: ({ alpha: number; norma_bobot: number } & Record<string, number>)[];
    kolom_identik_dibuang: { dibuang: string; identik_dengan: string }[];
  };
  model_terbaik: string;
  pemilihan_model: {
    aturan: string;
    model_rmse_terkecil: string;
    rmse_terkecil: number;
    standard_error: number;
    batas_1se: number;
    kandidat: { model: string; rmse_cv: number; kompleksitas: number; dalam_batas_1se: boolean }[];
    model_terpilih: string;
  };
  penyesuaian_harga: PenyesuaianHarga;
}

export interface PenyesuaianHarga {
  tahun_dasar_data: number;
  alasan_tahun_dasar: string;
  hicp_rata_rata_tahun_dasar: number;
  hicp_terkini: number;
  periode_hicp_terkini: string;
  sumber_hicp: string;
  faktor_inflasi: number;
  faktor_inflasi_diubah_manual: boolean;
  kurs_eur_idr: number;
  kurs_diubah_manual: boolean;
  sumber_kurs: string;
  tanggal_kurs: string;
}

export interface Statistik {
  jumlah: number;
  rata_rata: number;
  std: number;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  skewness: number;
}

export interface RingkasanDataset {
  nama_file: string;
  jumlah_baris_mentah: number;
  jumlah_kolom_mentah: number;
  kolom_mentah: string[];
  jumlah_baris_dipakai: number;
  baris_dibuang: { company: string; product: string; alasan: string }[];
  nilai_kosong: number;
  baris_duplikat: number;
  jumlah_produk_unik: number;
  statistik_numerik: ({ kolom: string; label: string } & Statistik)[];
  statistik_log_harga: Statistik;
  jumlah_per_kategori: Record<string, { kategori: string; jumlah: number }[]>;
  contoh_baris: Record<string, string | number>[];
}

export interface HargaGrup {
  grup: string;
  jumlah: number;
  rata_rata: number;
  median: number;
  q1: number;
  q3: number;
  min: number;
  max: number;
}

export interface Eda {
  distribusi_harga: BinHistogram[];
  distribusi_log_harga: BinHistogram[];
  skewness: { harga: number; log_harga: number };
  korelasi: { kolom: string[]; nilai: number[][] };
  korelasi_dengan_log_harga: { fitur: string; label: string; korelasi: number }[];
  harga_per_merek: HargaGrup[];
  harga_per_tipe: HargaGrup[];
  harga_per_cpu: HargaGrup[];
  harga_per_gpu: HargaGrup[];
  harga_per_os: HargaGrup[];
  harga_per_ram: HargaGrup[];
  sebaran: { ram: number; ppi: number; ssd: number; berat: number; harga: number; tipe: string; merek: string }[];
}

export interface ModelCpu {
  nilai: string;
  kategori: string;
  ghz: number;
  ghz_variasi: number[];
  jumlah: number;
}

export interface ModelGpu {
  nilai: string;
  merek: string;
  jumlah: number;
}

export interface Opsi {
  company: { nilai: string; jumlah: number }[];
  type_name: { nilai: string; jumlah: number }[];
  cpu_kategori: { nilai: string; jumlah: number }[];
  cpu_model: ModelCpu[];
  gpu_merek: { nilai: string; jumlah: number }[];
  gpu_model: ModelGpu[];
  os: { nilai: string; jumlah: number }[];
  resolusi: { lebar_px: number; tinggi_px: number; nama: string | null; jumlah: number }[];
  ram_gb: number[];
  ssd_gb: number[];
  hdd_gb: number[];
  flash_gb: number[];
  inches: number[];
  berat_tipikal: { per_tipe_inci: Record<string, number>; per_tipe: Record<string, number>; semua: number };
  rentang: Record<string, { min: number; max: number }>;
  nilai_default: InputPrediksi;
  penyesuaian_harga: PenyesuaianHarga;
  kategori_model: Record<string, string[]> | null;
}

export interface InputPrediksi {
  company: string;
  type_name: string;
  inches: number;
  ram_gb: number;
  berat_kg: number;
  cpu_model?: string | null;
  cpu_kategori: string;
  cpu_ghz: number;
  lebar_px: number;
  tinggi_px: number;
  touchscreen: boolean;
  ips: boolean;
  ssd_gb: number;
  hdd_gb: number;
  flash_gb: number;
  gpu_model?: string | null;
  gpu_merek: string;
  os: string;
}

/** Laptop dari dataset (GET /api/dataset/laptop), untuk mengisi form sekali klik. */
export interface LaptopDataset {
  id: number;
  label: string;
  ringkas: string;
  harga_euro: number;
  set: "latih" | "uji";
  input: InputPrediksi;
}

export interface HasilCariLaptop {
  total_cocok: number;
  data: LaptopDataset[];
}

export interface HasilPrediksi {
  prediksi_id: number;
  model: InfoModel;
  harga_dataset_eur: number;
  rentang_dataset_eur: [number, number];
  harga_sekarang_eur: number;
  rentang_sekarang_eur: [number, number];
  harga_idr: number;
  rentang_idr: [number, number];
  prediksi_log: number;
  rentang: { metode: string; z: number; rmse_log_cv: number };
  kontribusi: {
    intersep_log: number;
    harga_acuan_euro: number;
    fitur: { fitur: string; label: string; kontribusi_log: number; efek_persen: number }[];
    acuan: { kategori: Record<string, string>; keterangan: string };
  };
  peringatan: string[];
  penyesuaian_harga: PenyesuaianHarga;
  fitur_turunan: { PPI: number };
}

export interface RiwayatPrediksi {
  total: number;
  data: {
    id: number;
    model_version_id: number;
    label_model: string;
    input: InputPrediksi & { faktor_inflasi: number; kurs_eur_idr: number };
    harga_prediksi_eur: number;
    harga_prediksi_idr: number;
    dibuat_pada: string | null;
  }[];
}
