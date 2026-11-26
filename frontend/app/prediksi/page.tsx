"use client";

import { useState, type FormEvent, type ReactNode } from "react";
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
import type { HasilPrediksi, InputPrediksi, Opsi, RiwayatPrediksi } from "@/lib/tipe";

// ---------------------------------------------------------------------------
// Elemen form
// ---------------------------------------------------------------------------
const KELAS_INPUT =
  "w-full rounded-lg border border-sumbu bg-white px-3 py-2 text-sm focus:border-biru focus:outline-none focus:ring-2 focus:ring-biru/20";

function Bidang({ label, bantuan, children }: { label: string; bantuan?: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1 block text-sm font-medium">{label}</span>
      {children}
      {bantuan && <span className="mt-1 block text-xs text-redup">{bantuan}</span>}
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
          {v === 0 && labelNol ? labelNol : `${formatAngka(v, 0)} ${satuan}`}
        </option>
      ))}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Form spesifikasi
// ---------------------------------------------------------------------------
function FormPrediksi({
  opsi,
  onHasil,
}: {
  opsi: Opsi;
  onHasil: (h: HasilPrediksi) => void;
}) {
  const [form, setForm] = useState<InputPrediksi>(opsi.nilai_default);
  const [mengirim, setMengirim] = useState(false);
  const [galat, setGalat] = useState<GalatApi | null>(null);

  const ubah = <K extends keyof InputPrediksi>(kunci: K, nilai: InputPrediksi[K]) =>
    setForm((f) => ({ ...f, [kunci]: nilai }));

  const r = opsi.rentang;
  const tanpaNol = (daftar: number[]) => Array.from(new Set([0, ...daftar])).sort((a, b) => a - b);

  async function kirim(e: FormEvent) {
    e.preventDefault();
    setMengirim(true);
    setGalat(null);
    try {
      onHasil(await kirimJson<HasilPrediksi>("/api/predict", form));
    } catch (err) {
      setGalat(err instanceof GalatApi ? err : new GalatApi(String(err), -1));
    } finally {
      setMengirim(false);
    }
  }

  return (
    <form onSubmit={kirim} className="space-y-5">
      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">Umum</legend>
        <Bidang label="Merek">
          <PilihTeks nilai={form.company} opsi={opsi.company} onUbah={(v) => ubah("company", v)} />
        </Bidang>
        <Bidang label="Tipe laptop">
          <PilihTeks nilai={form.type_name} opsi={opsi.type_name} onUbah={(v) => ubah("type_name", v)} />
        </Bidang>
        <Bidang label="Sistem operasi">
          <PilihTeks nilai={form.os} opsi={opsi.os} onUbah={(v) => ubah("os", v)} />
        </Bidang>
        <Bidang label="Berat (kg)" bantuan={`Data latih: ${formatAngka(r.Weight.min, 2)}–${formatAngka(r.Weight.max, 2)} kg`}>
          <input
            type="number"
            className={KELAS_INPUT}
            step={0.01}
            min={0.31}
            max={6}
            required
            value={form.berat_kg}
            onChange={(e) => ubah("berat_kg", Number(e.target.value))}
          />
        </Bidang>
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">Prosesor, memori & grafis</legend>
        <Bidang label="Kategori CPU">
          <PilihTeks nilai={form.cpu_kategori} opsi={opsi.cpu_kategori} onUbah={(v) => ubah("cpu_kategori", v)} />
        </Bidang>
        <Bidang label="Frekuensi CPU (GHz)" bantuan={`Data latih: ${formatAngka(r.CPU_GHz.min, 1)}–${formatAngka(r.CPU_GHz.max, 1)} GHz`}>
          <input
            type="number"
            className={KELAS_INPUT}
            step={0.1}
            min={0.5}
            max={6}
            required
            value={form.cpu_ghz}
            onChange={(e) => ubah("cpu_ghz", Number(e.target.value))}
          />
        </Bidang>
        <Bidang label="RAM">
          <PilihAngka nilai={form.ram_gb} opsi={opsi.ram_gb} onUbah={(v) => ubah("ram_gb", v)} satuan="GB" />
        </Bidang>
        <Bidang label="Merek GPU">
          <PilihTeks nilai={form.gpu_merek} opsi={opsi.gpu_merek} onUbah={(v) => ubah("gpu_merek", v)} />
        </Bidang>
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-3">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">Penyimpanan (minimal satu diisi)</legend>
        <Bidang label="SSD">
          <PilihAngka nilai={form.ssd_gb} opsi={tanpaNol(opsi.ssd_gb)} onUbah={(v) => ubah("ssd_gb", v)} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
        <Bidang label="HDD">
          <PilihAngka nilai={form.hdd_gb} opsi={tanpaNol(opsi.hdd_gb)} onUbah={(v) => ubah("hdd_gb", v)} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
        <Bidang label="Flash storage">
          <PilihAngka nilai={form.flash_gb} opsi={tanpaNol(opsi.flash_gb)} onUbah={(v) => ubah("flash_gb", v)} satuan="GB" labelNol="Tidak ada" />
        </Bidang>
      </fieldset>

      <fieldset className="grid gap-4 sm:grid-cols-2">
        <legend className="mb-2 text-sm font-semibold text-tinta-2">Layar</legend>
        <Bidang label="Ukuran layar (inci)" bantuan={`Data latih: ${formatAngka(r.Inches.min, 1)}–${formatAngka(r.Inches.max, 1)} inci`}>
          <input
            type="number"
            className={KELAS_INPUT}
            step={0.1}
            min={7}
            max={21}
            required
            list="daftar-inci"
            value={form.inches}
            onChange={(e) => ubah("inches", Number(e.target.value))}
          />
          <datalist id="daftar-inci">
            {opsi.inches.map((v) => (
              <option key={v} value={v} />
            ))}
          </datalist>
        </Bidang>
        <Bidang label="Resolusi">
          <select
            className={KELAS_INPUT}
            value={`${form.lebar_px}x${form.tinggi_px}`}
            onChange={(e) => {
              const [l, t] = e.target.value.split("x").map(Number);
              setForm((f) => ({ ...f, lebar_px: l, tinggi_px: t }));
            }}
          >
            {opsi.resolusi.map((o) => (
              <option key={`${o.lebar_px}x${o.tinggi_px}`} value={`${o.lebar_px}x${o.tinggi_px}`}>
                {o.lebar_px} × {o.tinggi_px} ({o.jumlah} laptop)
              </option>
            ))}
          </select>
        </Bidang>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.touchscreen} onChange={(e) => ubah("touchscreen", e.target.checked)} className="h-4 w-4 accent-biru" />
          Layar sentuh (touchscreen)
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={form.ips} onChange={(e) => ubah("ips", e.target.checked)} className="h-4 w-4 accent-biru" />
          Panel IPS
        </label>
      </fieldset>

      {galat && <PesanGalat galat={galat} />}

      <div className="flex flex-wrap gap-3">
        <button
          type="submit"
          disabled={mengirim}
          className="rounded-lg bg-biru px-5 py-2 font-medium text-white hover:bg-biru-tua disabled:opacity-60"
        >
          {mengirim ? "Menghitung…" : "Prediksi harga"}
        </button>
        <button
          type="button"
          onClick={() => setForm(opsi.nilai_default)}
          className="rounded-lg border border-sumbu bg-white px-4 py-2 text-sm hover:bg-halaman"
        >
          Kembalikan ke nilai umum
        </button>
      </div>
    </form>
  );
}

// ---------------------------------------------------------------------------
// Hasil prediksi
// ---------------------------------------------------------------------------
function HasilPanel({ hasil }: { hasil: HasilPrediksi }) {
  const p = hasil.penyesuaian_harga;
  const acuan = hasil.kontribusi.acuan.kategori;
  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-biru/30 bg-biru-muda p-5">
        <div className="text-sm text-tinta-2">Perkiraan harga setara hari ini</div>
        <div className="mt-1 text-3xl font-semibold sm:text-4xl">{formatRupiah(hasil.harga_idr)}</div>
        <div className="mt-2 text-sm text-tinta-2">
          Rentang perkiraan ~95%: <b className="text-tinta">{formatRupiahSingkat(hasil.rentang_idr[0])}</b> sampai{" "}
          <b className="text-tinta">{formatRupiahSingkat(hasil.rentang_idr[1])}</b>
        </div>
      </div>

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
              <b>3.</b> × kurs Rp{formatAngka(p.kurs_eur_idr, 0)}/€{" "}
              <span className="text-xs text-redup">({p.tanggal_kurs})</span>
            </span>
            <span className="angka font-medium">{formatRupiah(hasil.harga_idr)}</span>
          </li>
        </ol>
        <p className="mt-2 text-xs text-tinta-2">
          Penyesuaian inflasi menyetarakan nilai uang, bukan harga pasar laptop baru: laptop dengan spesifikasi setara data{" "}
          {p.tahun_dasar_data} saat ini umumnya dijual lebih murah karena teknologinya sudah lama. Rentang dihitung dari{" "}
          {hasil.rentang.metode} (RMSE log = {formatAngka(hasil.rentang.rmse_log_cv, 3)}).
        </p>
        <details className="mt-2 text-xs text-redup">
          <summary className="cursor-pointer text-tinta-2">Sumber data penyesuaian harga</summary>
          <p className="mt-1">Inflasi: {p.sumber_hicp}</p>
          <p className="mt-1">Kurs: {p.sumber_kurs}, {p.tanggal_kurs}</p>
          <p className="mt-1">Tahun dasar: {p.alasan_tahun_dasar}</p>
        </details>
      </div>

      <div>
        <h3 className="text-sm font-semibold">Kontribusi tiap fitur</h3>
        <p className="mb-2 mt-1 text-xs text-tinta-2">
          Titik awal adalah <b>laptop acuan</b> seharga {formatEuro(hasil.kontribusi.harga_acuan_euro)} (tingkat harga dataset):{" "}
          {Object.entries(acuan)
            .map(([k, v]) => `${k} = ${v}`)
            .join(", ")}
          , fitur numerik bernilai rata-rata data latih. Batang menunjukkan berapa persen tiap fitur menaikkan (biru) atau
          menurunkan (merah) harga dibanding acuan tersebut. Contoh: “tanpa HDD” bisa bernilai negatif karena laptop rata-rata
          di data latih memiliki HDD. Fitur yang nilainya sama dengan acuan (efek 0%) tidak ditampilkan.
        </p>
        <GrafikBatangH
          divergen
          lebarLabel={170}
          formatNilai={(v) => formatPersen(v, 1, true)}
          data={hasil.kontribusi.fitur.filter((f) => Math.abs(f.kontribusi_log) > 1e-9).map((f) => ({
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
            <th>CPU / RAM</th>
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
                  {i.cpu_kategori} {i.cpu_ghz} GHz
                  <div className="text-xs text-redup">
                    RAM {i.ram_gb} GB · GPU {i.gpu_merek}
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
// Halaman
// ---------------------------------------------------------------------------
export default function HalamanPrediksi() {
  const opsi = useApi<Opsi>("/api/options");
  const riwayat = useApi<RiwayatPrediksi>("/api/predictions/history?batas=10");
  const [hasil, setHasil] = useState<HasilPrediksi | null>(null);

  return (
    <div className="space-y-6">
      <JudulHalaman
        judul="Prediksi Harga"
        keterangan="Isi spesifikasi laptop. Pilihan di setiap dropdown diambil dari dataset, beserta jumlah laptop yang memilikinya."
      />

      {!opsi.data ? (
        <StatusApi memuat={opsi.memuat} galat={opsi.galat} muatUlang={opsi.muatUlang} />
      ) : (
        <div className="grid gap-6 xl:grid-cols-12">
          <div className="xl:col-span-5">
            <Kartu judul="Spesifikasi laptop">
              <FormPrediksi
                opsi={opsi.data}
                onHasil={(h) => {
                  setHasil(h);
                  riwayat.muatUlang();
                }}
              />
            </Kartu>
          </div>
          <div className="xl:col-span-7">
            <Kartu
              judul="Hasil prediksi"
              aksi={hasil && <Lencana warna="biru">{hasil.model.label}</Lencana>}
            >
              {hasil ? (
                <HasilPanel hasil={hasil} />
              ) : (
                <p className="text-sm text-tinta-2">
                  Hasil akan tampil di sini setelah Anda menekan <b>Prediksi harga</b>. Nilai awal form adalah konfigurasi yang
                  paling umum di dataset.
                </p>
              )}
            </Kartu>
          </div>
        </div>
      )}

      <Kartu
        judul="Riwayat prediksi"
        keterangan="10 prediksi terakhir yang tersimpan di tabel predictions (database)."
        aksi={riwayat.data && <span className="text-sm text-tinta-2">Total: {riwayat.data.total}</span>}
      >
        {riwayat.data ? (
          <TabelRiwayat riwayat={riwayat.data} />
        ) : (
          <StatusApi memuat={riwayat.memuat} galat={riwayat.galat} muatUlang={riwayat.muatUlang} />
        )}
      </Kartu>
    </div>
  );
}
