"use client";

/**
 * Halaman Prediksi Harga.
 *
 * Dibuat agar mudah dipakai orang awam:
 * - Bisa mulai dari laptop yang sudah ada di dataset (cari → klik "Pakai").
 * - Prosesor & kartu grafis dipilih berdasarkan NAMA MODEL (seperti yang tertulis
 *   di spesifikasi laptop); kategori CPU, clock GHz, dan merek GPU terisi otomatis.
 * - Berat terisi otomatis dari berat tipikal laptop sejenis (tetap bisa diubah).
 * - Jika laptop diambil dari dataset, hasil prediksi dibandingkan dengan harga aslinya.
 */

import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { GrafikBatangH } from "@/components/grafik";
import { JudulHalaman, Kartu, KotakInfo, Lencana, PesanGalat, StatusApi } from "@/components/ui";
import { GalatApi, kirimJson, useApi } from "@/lib/api";
import {
  formatAngka,
  formatEuro,
  formatPersen,
  formatRupiah,
  formatRupiahSingkat,
  formatTanggal,
} from "@/lib/format";
import type {
  HasilCariLaptop,
  HasilPrediksi,
  InputPrediksi,
  LaptopDataset,
  ModelCpu,
  ModelGpu,
  Opsi,
  RiwayatPrediksi,
} from "@/lib/tipe";

// ---------------------------------------------------------------------------
// Elemen form
// ---------------------------------------------------------------------------
const KELAS_INPUT =
  "w-full rounded-lg border border-sumbu bg-white px-3 py-2 text-sm focus:border-biru focus:outline-none focus:ring-2 focus:ring-biru/20";

function Bidang({ label, bantuan, children }: { label: string; bantuan?: ReactNode; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      {children}
      {bantuan && <span className="mt-1 block text-xs text-tinta-2">{bantuan}</span>}
    </label>
  );
}

function PilihTeks({
  nilai,
  opsi,
  onUbah,
}: {
  nilai: string;
  opsi: { nilai: string; jumlah: number }[];
  onUbah: (v: string) => void;
}) {
  return (
    <select className={KELAS_INPUT} value={nilai} onChange={(e) => onUbah(e.target.value)}>
      {opsi.map((o) => (
        <option key={o.nilai} value={o.nilai}>
          {o.nilai} ({o.jumlah} laptop)
        </option>
      ))}
    </select>
  );
}

function PilihAngka({
  nilai,
  opsi,
  onUbah,
  satuan,
  labelNol,
}: {
  nilai: number;
  opsi: number[];
  onUbah: (v: number) => void;
  satuan: string;
  labelNol?: string;
}) {
  return (
    <select className={KELAS_INPUT} value={nilai} onChange={(e) => onUbah(Number(e.target.value))}>
      {opsi.map((v) => (
        <option key={v} value={v}>
          {v === 0 && labelNol ? labelNol : `${formatAngka(v, v % 1 === 0 ? 0 : 1)} ${satuan}`}
        </option>
      ))}
    </select>
  );
}

/** Mengelompokkan daftar berdasarkan kunci, urutan kelompok mengikuti kemunculan pertama. */
function kelompokkan<T>(daftar: T[], kunci: (x: T) => string): [string, T[]][] {
  const peta = new Map<string, T[]>();
  for (const x of daftar) peta.set(kunci(x), [...(peta.get(kunci(x)) ?? []), x]);
  return Array.from(peta.entries());
}

function beratTipikal(opsi: Opsi, tipe: string, inci: number): number {
  const b = opsi.berat_tipikal;
  return b.per_tipe_inci[`${tipe}|${inci}`] ?? b.per_tipe[tipe] ?? b.semua;
}

// ---------------------------------------------------------------------------
// Pencarian laptop di dataset
// ---------------------------------------------------------------------------
function CariLaptop({ onPakai, idTerpakai }: { onPakai: (l: LaptopDataset) => void; idTerpakai: number | null }) {
  const [ketikan, setKetikan] = useState("");
  const [kataKunci, setKataKunci] = useState("");
  // tunggu 300 ms setelah berhenti mengetik, baru minta ke API
  useEffect(() => {
    const t = setTimeout(() => setKataKunci(ketikan.trim()), 300);
    return () => clearTimeout(t);
  }, [ketikan]);
  const hasil = useApi<HasilCariLaptop>(`/api/dataset/laptop?batas=8&q=${encodeURIComponent(kataKunci)}`);
  const data = hasil.data ?? hasil.terakhir?.data;

  return (
    <div className="space-y-3">
      <input
        type="search"
        value={ketikan}
        onChange={(e) => setKetikan(e.target.value)}
        placeholder="Ketik merek, seri, atau prosesor, mis. “dell xps”, “asus gaming”, “i7 16gb”"
        className={KELAS_INPUT}
        aria-label="Cari laptop di dataset"
      />
      {hasil.galat && <PesanGalat galat={hasil.galat} onCobaLagi={hasil.muatUlang} />}
      {data && (
        <>
          <p className="text-xs text-tinta-2">
            {kataKunci
              ? `${data.total_cocok} laptop cocok${data.total_cocok > data.data.length ? `, ditampilkan ${data.data.length}` : ""}.`
              : "Contoh laptop dari data uji (belum pernah dilihat model), dari yang termurah sampai termahal:"}
          </p>
          <ul className="grid grid-cols-1 gap-2 md:grid-cols-2">
            {data.data.map((l) => (
              <li
                key={l.id}
                className={`flex items-start justify-between gap-3 rounded-lg border p-3 ${
                  l.id === idTerpakai ? "border-biru bg-biru-muda" : "border-garis bg-white"
                }`}
              >
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium" title={l.label}>
                    {l.label}
                  </div>
                  <div className="text-xs text-tinta-2">{l.ringkas}</div>
                  <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-tinta-2">
                      Harga asli: <b className="text-tinta">{formatEuro(l.harga_euro)}</b>
                    </span>
                    <Lencana warna={l.set === "uji" ? "hijau" : "abu"}>data {l.set}</Lencana>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onPakai(l)}
                  className="shrink-0 rounded-lg border border-biru px-3 py-1 text-sm font-medium text-biru-tua hover:bg-biru-muda"
                >
                  {l.id === idTerpakai ? "Terpakai" : "Pakai"}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {!data && !hasil.galat && <p className="text-sm text-tinta-2">Memuat daftar laptop…</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Form spesifikasi
// ---------------------------------------------------------------------------
function FormSpesifikasi({
  opsi,
  form,
  setForm,
  beratManual,
  setBeratManual,
  mengirim,
  galat,
  onKirim,
  onReset,
}: {
  opsi: Opsi;
  form: InputPrediksi;
  setForm: (f: InputPrediksi) => void;
  beratManual: boolean;
  setBeratManual: (v: boolean) => void;
  mengirim: boolean;
  galat: GalatApi | null;
  onKirim: (e: FormEvent) => void;
  onReset: () => void;
}) {
  const cpu: ModelCpu | undefined = opsi.cpu_model.find((c) => c.nilai === form.cpu_model);
  const tanpaNol = (daftar: number[]) => Array.from(new Set([0, ...daftar])).sort((a, b) => a - b);

  // Mengubah tipe/ukuran layar ikut memperbarui berat tipikal (kecuali berat diisi manual)
  const ubahUmum = (perubahan: Partial<InputPrediksi>) => {
    const baru = { ...form, ...perubahan };
    if (!beratManual) baru.berat_kg = beratTipikal(opsi, baru.type_name, baru.inches);
    setForm(baru);
  };
  const pilihCpu = (nama: string) => {
    const m = opsi.cpu_model.find((c) => c.nilai === nama)!;
    setForm({ ...form, cpu_model: m.nilai, cpu_kategori: m.kategori, cpu_ghz: m.ghz });
  };
  const pilihGpu = (nama: string) => {
    const g: ModelGpu = opsi.gpu_model.find((x) => x.nilai === nama)!;
    setForm({ ...form, gpu_model: g.nilai, gpu_merek: g.merek });
  };
  const berat = beratTipikal(opsi, form.type_name, form.inches);
  const kosongPenyimpanan = form.ssd_gb + form.hdd_gb + form.flash_gb <= 0;

  return (
    <form onSubmit={onKirim} className="space-y-6">
      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">1. Laptop</legend>
        <Bidang label="Merek">
          <PilihTeks nilai={form.company} opsi={opsi.company} onUbah={(v) => ubahUmum({ company: v })} />
        </Bidang>
        <Bidang label="Tipe laptop" bantuan="Notebook = laptop biasa; Ultrabook = tipis & ringan; Gaming = GPU kuat.">
          <PilihTeks nilai={form.type_name} opsi={opsi.type_name} onUbah={(v) => ubahUmum({ type_name: v })} />
        </Bidang>
        <Bidang label="Sistem operasi">
          <PilihTeks nilai={form.os} opsi={opsi.os} onUbah={(v) => ubahUmum({ os: v })} />
        </Bidang>
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">2. Prosesor, RAM & kartu grafis</legend>
        <div className="sm:col-span-2">
          <Bidang
            label="Prosesor (CPU)"
            bantuan={
              cpu && (
                <>
                  Terisi otomatis: kategori <b className="text-tinta">{cpu.kategori}</b>, clock{" "}
                  <b className="text-tinta">{formatAngka(form.cpu_ghz, 1)} GHz</b>. Lihat nama prosesor di stiker atau
                  spesifikasi laptop, mis. “Core i5 7200U”.
                </>
              )
            }
          >
            <select className={KELAS_INPUT} value={form.cpu_model ?? ""} onChange={(e) => pilihCpu(e.target.value)}>
              {kelompokkan(opsi.cpu_model, (c) => c.kategori).map(([kategori, daftar]) => (
                <optgroup key={kategori} label={kategori}>
                  {daftar.map((c) => (
                    <option key={c.nilai} value={c.nilai}>
                      {c.nilai} · {formatAngka(c.ghz, 1)} GHz ({c.jumlah} laptop)
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </Bidang>
        </div>
        {cpu && cpu.ghz_variasi.length > 1 && (
          <Bidang label="Clock dasar CPU" bantuan="Model ini muncul dengan beberapa clock di dataset; yang paling umum dipilih otomatis.">
            <PilihAngka nilai={form.cpu_ghz} opsi={cpu.ghz_variasi} onUbah={(v) => setForm({ ...form, cpu_ghz: v })} satuan="GHz" />
          </Bidang>
        )}
        <Bidang label="RAM">
          <PilihAngka nilai={form.ram_gb} opsi={opsi.ram_gb} onUbah={(v) => setForm({ ...form, ram_gb: v })} satuan="GB" />
        </Bidang>
        <div className="sm:col-span-2">
          <Bidang
            label="Kartu grafis (GPU)"
            bantuan={
              <>
                Model hanya memakai <b className="text-tinta">merek</b> GPU ({form.gpu_merek}); tipenya belum menjadi fitur model.
              </>
            }
          >
            <select className={KELAS_INPUT} value={form.gpu_model ?? ""} onChange={(e) => pilihGpu(e.target.value)}>
              {kelompokkan(opsi.gpu_model, (g) => g.merek).map(([merek, daftar]) => (
                <optgroup key={merek} label={merek}>
                  {daftar.map((g) => (
                    <option key={g.nilai} value={g.nilai}>
                      {g.nilai} ({g.jumlah} laptop)
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </Bidang>
        </div>
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-3">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">3. Penyimpanan (isi minimal satu)</legend>
        <Bidang label="SSD">
          <PilihAngka nilai={form.ssd_gb} opsi={tanpaNol(opsi.ssd_gb)} onUbah={(v) => setForm({ ...form, ssd_gb: v })} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
        <Bidang label="HDD">
          <PilihAngka nilai={form.hdd_gb} opsi={tanpaNol(opsi.hdd_gb)} onUbah={(v) => setForm({ ...form, hdd_gb: v })} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
        <Bidang label="Flash (eMMC)">
          <PilihAngka nilai={form.flash_gb} opsi={tanpaNol(opsi.flash_gb)} onUbah={(v) => setForm({ ...form, flash_gb: v })} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
        {kosongPenyimpanan && <p className="text-xs text-merah sm:col-span-3">Pilih minimal satu jenis penyimpanan.</p>}
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">4. Layar & berat</legend>
        <Bidang label="Ukuran layar">
          <PilihAngka nilai={form.inches} opsi={opsi.inches} onUbah={(v) => ubahUmum({ inches: v })} satuan="inci" />
        </Bidang>
        <Bidang label="Resolusi">
          <select
            className={KELAS_INPUT}
            value={`${form.lebar_px}x${form.tinggi_px}`}
            onChange={(e) => {
              const [l, t] = e.target.value.split("x").map(Number);
              setForm({ ...form, lebar_px: l, tinggi_px: t });
            }}
          >
            {opsi.resolusi.map((o) => (
              <option key={`${o.lebar_px}x${o.tinggi_px}`} value={`${o.lebar_px}x${o.tinggi_px}`}>
                {o.nama ? `${o.nama} — ` : ""}
                {o.lebar_px} × {o.tinggi_px} ({o.jumlah} laptop)
              </option>
            ))}
          </select>
        </Bidang>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.touchscreen} onChange={(e) => setForm({ ...form, touchscreen: e.target.checked })} className="h-4 w-4 accent-biru" />
          Layar sentuh (touchscreen)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.ips} onChange={(e) => setForm({ ...form, ips: e.target.checked })} className="h-4 w-4 accent-biru" />
          Panel IPS (warna & sudut pandang lebih baik)
        </label>
        <div className="sm:col-span-2">
          <Bidang
            label="Berat (kg)"
            bantuan={
              beratManual ? (
                <>
                  Diisi manual.{" "}
                  <button
                    type="button"
                    className="text-biru-tua underline"
                    onClick={() => {
                      setBeratManual(false);
                      setForm({ ...form, berat_kg: berat });
                    }}
                  >
                    Pakai berat tipikal ({formatAngka(berat, 2)} kg)
                  </button>
                </>
              ) : (
                <>Terisi otomatis: berat tipikal {form.type_name} {formatAngka(form.inches, 1)} inci di dataset. Ubah jika Anda tahu beratnya.</>
              )
            }
          >
            <input
              type="number"
              className={KELAS_INPUT}
              step={0.01}
              min={0.31}
              max={6}
              required
              value={form.berat_kg}
              onChange={(e) => {
                setBeratManual(true);
                setForm({ ...form, berat_kg: Number(e.target.value) });
              }}
            />
          </Bidang>
        </div>
      </fieldset>

      {galat && <PesanGalat galat={galat} />}

      <div className="flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={mengirim || kosongPenyimpanan}
          className="rounded-lg bg-biru px-5 py-2 font-medium text-white hover:bg-biru-tua disabled:opacity-60"
        >
          {mengirim ? "Menghitung…" : "Prediksi harga"}
        </button>
        <button type="button" onClick={onReset} className="rounded-lg border border-sumbu bg-white px-4 py-2 text-sm hover:bg-halaman">
          Kembalikan ke laptop paling umum
        </button>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Hasil prediksi
// ---------------------------------------------------------------------------
function PerbandinganHargaAsli({ hasil, laptop }: { hasil: HasilPrediksi; laptop: LaptopDataset }) {
  const selisih = ((hasil.harga_dataset_eur - laptop.harga_euro) / laptop.harga_euro) * 100;
  const diDalam = laptop.harga_euro >= hasil.rentang_dataset_eur[0] && laptop.harga_euro <= hasil.rentang_dataset_eur[1];
  return (
    <div className="rounded-lg border border-garis bg-white p-4 text-sm">
      <div className="mb-2 flex flex-wrap items-center gap-2 font-semibold">
        Dibandingkan harga asli: {laptop.label}
        <Lencana warna={laptop.set === "uji" ? "hijau" : "abu"}>data {laptop.set}</Lencana>
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <div className="text-xs text-tinta-2">Harga asli (dataset)</div>
          <div className="angka text-lg font-semibold">{formatEuro(laptop.harga_euro)}</div>
        </div>
        <div>
          <div className="text-xs text-tinta-2">Prediksi model</div>
          <div className="angka text-lg font-semibold">{formatEuro(hasil.harga_dataset_eur)}</div>
        </div>
        <div>
          <div className="text-xs text-tinta-2">Selisih</div>
          <div className={`angka text-lg font-semibold ${Math.abs(selisih) <= 20 ? "text-baik" : "text-merah"}`}>
            {formatPersen(selisih, 1, true)}
          </div>
        </div>
      </div>
      <p className="mt-2 text-xs text-tinta-2">
        Harga asli {diDalam ? "berada di dalam" : "berada di luar"} rentang perkiraan model.{" "}
        {laptop.set === "uji"
          ? "Laptop ini termasuk data uji: tidak dipakai saat melatih model, jadi ini perbandingan yang jujur."
          : "Laptop ini termasuk data latih (sudah dipakai melatih model), jadi bukan uji yang sepenuhnya adil. Pilih laptop bertanda “data uji” untuk menguji model."}
      </p>
    </div>
  );
}

function HasilPanel({ hasil, laptop }: { hasil: HasilPrediksi; laptop: LaptopDataset | null }) {
  const p = hasil.penyesuaian_harga;
  const acuan = hasil.kontribusi.acuan.kategori;
  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-biru/30 bg-biru-muda p-5">
        <div className="text-sm text-tinta-2">Perkiraan harga setara hari ini</div>
        <div className="mt-1 text-3xl font-semibold sm:text-4xl">{formatRupiahSingkat(hasil.harga_idr)}</div>
        <div className="angka text-sm text-tinta-2">{formatRupiah(hasil.harga_idr)}</div>
        <div className="mt-2 text-sm text-tinta-2">
          Kisaran wajar (~95%): <b className="text-tinta">{formatRupiahSingkat(hasil.rentang_idr[0])}</b> sampai{" "}
          <b className="text-tinta">{formatRupiahSingkat(hasil.rentang_idr[1])}</b>
        </div>
      </div>

      {laptop && <PerbandinganHargaAsli hasil={hasil} laptop={laptop} />}

      {hasil.peringatan.length > 0 && (
        <KotakInfo jenis="peringatan" judul="Perhatikan sebelum memakai angka ini">
          <ul className="list-disc space-y-1 pl-4">
            {hasil.peringatan.map((p, i) => (
              <li key={i}>{p}</li>
            ))}
          </ul>
        </KotakInfo>
      )}

      <div>
        <h3 className="mb-2 text-sm font-semibold">Dari mana angka ini?</h3>
        <ol className="space-y-2 text-sm">
          <li className="flex flex-wrap justify-between gap-2 rounded-lg border border-garis bg-white p-3">
            <span>
              <b>1.</b> Prediksi model pada tingkat harga dataset (sekitar {p.tahun_dasar_data})
            </span>
            <span className="angka font-medium">
              {formatEuro(hasil.harga_dataset_eur)}{" "}
              <span className="text-xs text-redup">
                ({formatEuro(hasil.rentang_dataset_eur[0])} – {formatEuro(hasil.rentang_dataset_eur[1])})
              </span>
            </span>
          </li>
          <li className="flex flex-wrap justify-between gap-2 rounded-lg border border-garis bg-white p-3">
            <span>
              <b>2.</b> × faktor inflasi zona euro {formatAngka(p.faktor_inflasi, 3)}{" "}
              <span className="text-xs text-redup">
                (HICP {formatAngka(p.hicp_terkini, 2)} pada {p.periode_hicp_terkini} ÷ rata-rata {p.tahun_dasar_data}{" "}
                {formatAngka(p.hicp_rata_rata_tahun_dasar, 2)})
              </span>
            </span>
            <span className="angka font-medium">{formatEuro(hasil.harga_sekarang_eur)}</span>
          </li>
          <li className="flex flex-wrap justify-between gap-2 rounded-lg border border-garis bg-white p-3">
            <span>
              <b>3.</b> × kurs Rp{formatAngka(p.kurs_eur_idr, 0)}/€ <span className="text-xs text-redup">({p.tanggal_kurs})</span>
            </span>
            <span className="angka font-medium">{formatRupiah(hasil.harga_idr)}</span>
          </li>
        </ol>
        <p className="mt-2 text-xs text-tinta-2">
          Penyesuaian inflasi menyetarakan nilai uang, bukan harga pasar laptop baru: laptop dengan spesifikasi setara data{" "}
          {p.tahun_dasar_data} saat ini umumnya dijual lebih murah karena teknologinya sudah lama. Kisaran dihitung dari{" "}
          {hasil.rentang.metode} (RMSE log = {formatAngka(hasil.rentang.rmse_log_cv, 3)}).
        </p>
        <details className="mt-2 text-xs text-redup">
          <summary className="cursor-pointer text-tinta-2">Sumber data penyesuaian harga</summary>
          <p className="mt-1">Inflasi: {p.sumber_hicp}</p>
          <p className="mt-1">
            Kurs: {p.sumber_kurs}, {p.tanggal_kurs}
          </p>
          <p className="mt-1">Tahun dasar: {p.alasan_tahun_dasar}</p>
        </details>
      </div>

      <div>
        <h3 className="text-sm font-semibold">Fitur apa yang membuat harganya naik atau turun?</h3>
        <p className="mb-2 mt-1 text-xs text-tinta-2">
          Titik awal adalah <b>laptop acuan</b> seharga {formatEuro(hasil.kontribusi.harga_acuan_euro)} (tingkat harga dataset):{" "}
          {Object.entries(acuan)
            .map(([k, v]) => `${k} = ${v}`)
            .join(", ")}
          , fitur numerik bernilai rata-rata data latih. Batang biru menaikkan harga, merah menurunkan, dibanding acuan
          tersebut. Contoh: “tanpa HDD” bisa bernilai negatif karena laptop rata-rata di data latih memiliki HDD. Fitur yang
          nilainya sama dengan acuan (efek 0%) tidak ditampilkan.
        </p>
        <GrafikBatangH
          divergen
          lebarLabel={170}
          formatNilai={(v) => formatPersen(v, 1, true)}
          data={hasil.kontribusi.fitur
            .filter((f) => Math.abs(f.kontribusi_log) > 1e-9)
            .map((f) => ({
              label: f.label,
              nilai: f.efek_persen,
              info: [["Kontribusi ln(harga)", formatAngka(f.kontribusi_log, 4)]],
            }))}
        />
      </div>

      <p className="text-xs text-redup">
        Model: {hasil.model.label} · versi {hasil.model.nomor_versi} · id prediksi #{hasil.prediksi_id} · PPI layar dihitung{" "}
        {formatAngka(hasil.fitur_turunan.PPI, 1)}
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Riwayat
// ---------------------------------------------------------------------------
function TabelRiwayat({ riwayat }: { riwayat: RiwayatPrediksi }) {
  if (riwayat.data.length === 0) return <p className="text-sm text-tinta-2">Belum ada prediksi.</p>;
  return (
    <div className="overflow-x-auto">
      <table className="tabel">
        <thead>
          <tr>
            <th>Waktu</th>
            <th>Laptop</th>
            <th>Prosesor / RAM / GPU</th>
            <th>Penyimpanan</th>
            <th className="kanan">Harga dataset</th>
            <th className="kanan">Harga sekarang (Rp)</th>
          </tr>
        </thead>
        <tbody>
          {riwayat.data.map((r) => {
            const i = r.input;
            const simpan = [
              i.ssd_gb ? `SSD ${i.ssd_gb} GB` : null,
              i.hdd_gb ? `HDD ${i.hdd_gb} GB` : null,
              i.flash_gb ? `Flash ${i.flash_gb} GB` : null,
            ]
              .filter(Boolean)
              .join(" + ");
            return (
              <tr key={r.id}>
                <td className="whitespace-nowrap text-tinta-2">{formatTanggal(r.dibuat_pada)}</td>
                <td>
                  {i.company} · {i.type_name}
                  <div className="text-xs text-redup">
                    {i.inches} inci · {i.lebar_px}×{i.tinggi_px} · {i.os}
                  </div>
                </td>
                <td>
                  {i.cpu_model ?? `${i.cpu_kategori} ${i.cpu_ghz} GHz`}
                  <div className="text-xs text-redup">
                    RAM {i.ram_gb} GB · {i.gpu_model ?? `GPU ${i.gpu_merek}`}
                  </div>
                </td>
                <td>{simpan}</td>
                <td className="kanan">{formatEuro(r.harga_prediksi_eur)}</td>
                <td className="kanan font-medium">{formatRupiah(r.harga_prediksi_idr)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Penghubung: pencarian + form + hasil
// ---------------------------------------------------------------------------
function Prediktor({ opsi, onTersimpan }: { opsi: Opsi; onTersimpan: () => void }) {
  const [form, setFormAsli] = useState<InputPrediksi>(opsi.nilai_default);
  const [beratManual, setBeratManual] = useState(false);
  // laptop dataset yang sedang dipakai; dilepas begitu pengguna mengubah spesifikasi
  const [laptop, setLaptop] = useState<LaptopDataset | null>(null);
  const [mengirim, setMengirim] = useState(false);
  const [galat, setGalat] = useState<GalatApi | null>(null);
  const [hasil, setHasil] = useState<{ prediksi: HasilPrediksi; laptop: LaptopDataset | null } | null>(null);

  const setForm = (f: InputPrediksi) => {
    setFormAsli(f);
    setLaptop(null);
  };

  async function kirim(e: FormEvent) {
    e.preventDefault();
    setMengirim(true);
    setGalat(null);
    try {
      const prediksi = await kirimJson<HasilPrediksi>("/api/predict", form);
      setHasil({ prediksi, laptop });
      onTersimpan();
    } catch (err) {
      setGalat(err instanceof GalatApi ? err : new GalatApi(String(err), -1));
    } finally {
      setMengirim(false);
    }
  }

  return (
    <div className="space-y-6">
      <Kartu
        judul="Cara cepat: mulai dari laptop yang sudah ada"
        keterangan="Cari laptop di dataset yang mirip dengan laptop Anda, klik Pakai, lalu ubah bagian yang berbeda. Atau langsung isi form di bawah."
      >
        <CariLaptop
          idTerpakai={laptop?.id ?? null}
          onPakai={(l) => {
            setFormAsli(l.input);
            setBeratManual(true);
            setLaptop(l);
          }}
        />
      </Kartu>

      <div className="grid gap-6 xl:grid-cols-12">
        <div className="xl:col-span-5">
          <Kartu
            judul="Spesifikasi laptop"
            aksi={laptop && <Lencana warna="biru">dari dataset: {laptop.label}</Lencana>}
          >
            <FormSpesifikasi
              opsi={opsi}
              form={form}
              setForm={setForm}
              beratManual={beratManual}
              setBeratManual={setBeratManual}
              mengirim={mengirim}
              galat={galat}
              onKirim={kirim}
              onReset={() => {
                setForm(opsi.nilai_default);
                setBeratManual(false);
              }}
            />
          </Kartu>
        </div>
        <div className="xl:col-span-7">
          <Kartu judul="Hasil prediksi" aksi={hasil && <Lencana warna="biru">{hasil.prediksi.model.label}</Lencana>}>
            {hasil ? (
              <HasilPanel hasil={hasil.prediksi} laptop={hasil.laptop} />
            ) : (
              <p className="text-sm text-tinta-2">
                Hasil akan tampil di sini setelah Anda menekan <b>Prediksi harga</b>. Form sudah terisi dengan konfigurasi
                laptop yang paling umum di dataset, jadi Anda bisa langsung mencobanya.
              </p>
            )}
          </Kartu>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Halaman
// ---------------------------------------------------------------------------
export default function HalamanPrediksi() {
  const opsi = useApi<Opsi>("/api/options");
  const riwayat = useApi<RiwayatPrediksi>("/api/predictions/history?batas=10");
  const dataRiwayat = riwayat.data ?? riwayat.terakhir?.data;

  return (
    <div className="space-y-6">
      <JudulHalaman
        judul="Prediksi Harga"
        keterangan="Pilih spesifikasi seperti yang tertulis di laptop (nama prosesor, kartu grafis, RAM, penyimpanan). Semua pilihan diambil dari dataset, beserta jumlah laptop yang memilikinya."
      />

      {!opsi.data ? (
        <StatusApi memuat={opsi.memuat} galat={opsi.galat} muatUlang={opsi.muatUlang} />
      ) : (
        <Prediktor opsi={opsi.data} onTersimpan={riwayat.muatUlang} />
      )}

      <Kartu
        judul="Riwayat prediksi"
        keterangan="10 prediksi terakhir yang tersimpan di tabel predictions (database)."
        aksi={dataRiwayat && <span className="text-sm text-tinta-2">Total: {dataRiwayat.total}</span>}
      >
        {dataRiwayat ? (
          <TabelRiwayat riwayat={dataRiwayat} />
        ) : (
          <StatusApi memuat={riwayat.memuat} galat={riwayat.galat} muatUlang={riwayat.muatUlang} />
        )}
      </Kartu>
    </div>
  );
}
