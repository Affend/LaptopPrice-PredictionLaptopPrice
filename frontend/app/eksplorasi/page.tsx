"use client";

import { useState } from "react";
import { GrafikBatangH, GrafikHistogram, GrafikSebar, HeatmapKorelasi } from "@/components/grafik";
import { JudulHalaman, Kartu, KotakInfo, StatTile, StatusApi } from "@/components/ui";
import { useApi } from "@/lib/api";
import { formatAngka, formatEuro } from "@/lib/format";
import type { Eda, HargaGrup, RingkasanDataset } from "@/lib/tipe";

const LABEL_KOLOM: Record<string, string> = {
  Inches: "Inci",
  Ram: "RAM",
  Weight: "Berat",
  CPU_GHz: "GHz CPU",
  Lebar_px: "Lebar px",
  Tinggi_px: "Tinggi px",
  PPI: "PPI",
  SSD_GB: "SSD",
  HDD_GB: "HDD",
  Flash_GB: "Flash",
  ln_harga: "ln(harga)",
};

const PENGELOMPOKAN: { kunci: keyof Eda; label: string }[] = [
  { kunci: "harga_per_merek", label: "Merek" },
  { kunci: "harga_per_tipe", label: "Tipe laptop" },
  { kunci: "harga_per_cpu", label: "Kategori CPU" },
  { kunci: "harga_per_gpu", label: "Merek GPU" },
  { kunci: "harga_per_os", label: "Sistem operasi" },
  { kunci: "harga_per_ram", label: "RAM" },
];

const SUMBU_SEBAR = [
  { kunci: "ram", label: "RAM (GB)", log: true },
  { kunci: "ppi", label: "Kerapatan piksel (PPI)", log: false },
  { kunci: "ssd", label: "SSD (GB)", log: false },
  { kunci: "berat", label: "Berat (kg)", log: false },
] as const;

function Pilihan<T extends string>({
  label,
  nilai,
  opsi,
  onUbah,
}: {
  label: string;
  nilai: T;
  opsi: { nilai: T; label: string }[];
  onUbah: (v: T) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-tinta-2">
      {label}
      <select
        value={nilai}
        onChange={(e) => onUbah(e.target.value as T)}
        className="rounded-lg border border-sumbu bg-white px-2 py-1 text-tinta"
      >
        {opsi.map((o) => (
          <option key={o.nilai} value={o.nilai}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function HalamanEksplorasi() {
  const ringkasan = useApi<RingkasanDataset>("/api/dataset/summary");
  const eda = useApi<Eda>("/api/dataset/eda");
  const [kelompok, setKelompok] = useState<keyof Eda>("harga_per_merek");
  const [sumbuX, setSumbuX] = useState<(typeof SUMBU_SEBAR)[number]["kunci"]>("ram");

  const judul = (
    <JudulHalaman
      judul="Eksplorasi Data"
      keterangan="Gambaran isi dataset sebelum dimodelkan. Semua angka dihitung backend langsung dari file CSV."
    />
  );
  if (!ringkasan.data || !eda.data) {
    return (
      <>
        {judul}
        <StatusApi
          memuat={ringkasan.memuat || eda.memuat}
          galat={ringkasan.galat ?? eda.galat}
          muatUlang={() => {
            ringkasan.muatUlang();
            eda.muatUlang();
          }}
        />
      </>
    );
  }
  const r = ringkasan.data;
  const e = eda.data;
  const harga = r.statistik_numerik.find((s) => s.kolom === "Price_euros")!;
  const grup = e[kelompok] as HargaGrup[];
  const sumbu = SUMBU_SEBAR.find((s) => s.kunci === sumbuX)!;

  return (
    <div className="space-y-6">
      {judul}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
        <StatTile label="Baris di CSV" nilai={formatAngka(r.jumlah_baris_mentah, 0)} catatan={r.nama_file} />
        <StatTile label="Baris dipakai" nilai={formatAngka(r.jumlah_baris_dipakai, 0)} catatan={`${r.baris_dibuang.length} dikeluarkan`} />
        <StatTile label="Kolom" nilai={r.jumlah_kolom_mentah} />
        <StatTile label="Nilai kosong" nilai={r.nilai_kosong} catatan={`duplikat penuh: ${r.baris_duplikat}`} />
        <StatTile label="Median harga" nilai={formatEuro(harga.median)} catatan={`rata-rata ${formatEuro(harga.rata_rata)}`} />
        <StatTile label="Rentang harga" nilai={<span className="text-lg">{formatEuro(harga.min)} – {formatEuro(harga.max)}</span>} />
      </div>

      {r.baris_dibuang.length > 0 && (
        <KotakInfo judul="Data yang dikeluarkan dari pemodelan">
          {r.baris_dibuang.map((b) => (
            <span key={b.product}>
              {b.company} {b.product}: {b.alasan}.{" "}
            </span>
          ))}
          Satu data tidak cukup untuk mempelajari efek kategorinya, jadi model dinyatakan berlaku untuk laptop berprosesor
          Intel/AMD.
        </KotakInfo>
      )}

      <Kartu
        judul="Distribusi harga: asli vs logaritma"
        keterangan={
          <>
            Harga asli miring ke kanan (skewness {formatAngka(e.skewness.harga, 2)}): banyak laptop murah, sedikit yang sangat
            mahal. Setelah diambil logaritmanya, distribusi hampir simetris (skewness {formatAngka(e.skewness.log_harga, 2)}).
            Inilah alasan model memprediksi <b>ln(harga)</b>, bukan harga langsung.
          </>
        }
      >
        <div className="grid gap-6 md:grid-cols-2">
          <div>
            <h3 className="mb-1 text-sm font-medium">Harga (euro)</h3>
            <GrafikHistogram data={e.distribusi_harga} labelX="Harga (euro)" formatX={(v) => formatAngka(v, 0)} />
          </div>
          <div>
            <h3 className="mb-1 text-sm font-medium">ln(harga)</h3>
            <GrafikHistogram data={e.distribusi_log_harga} labelX="ln(harga)" warna="#eb6834" />
          </div>
        </div>
      </Kartu>

      <Kartu
        judul="Korelasi dengan ln(harga)"
        keterangan="Koefisien korelasi Pearson tiap fitur numerik terhadap ln(harga). Biru = makin besar fitur, makin mahal; merah = sebaliknya."
      >
        <div className="max-w-3xl">
          <GrafikBatangH
            divergen
            lebarLabel={170}
            formatNilai={(v) => formatAngka(v, 2)}
            data={e.korelasi_dengan_log_harga.map((k) => ({ label: k.label, nilai: k.korelasi }))}
          />
        </div>
      </Kartu>

      <Kartu
        judul="Matriks korelasi antar-fitur"
        keterangan="Lebar, tinggi resolusi, dan PPI saling berkorelasi sangat tinggi (r ≥ 0,9). Ini tanda multikolinearitas, dibahas di halaman Validasi & Evaluasi."
      >
        <HeatmapKorelasi kolom={e.korelasi.kolom} nilai={e.korelasi.nilai} label={LABEL_KOLOM} />
      </Kartu>

      <Kartu
        judul="Harga per kelompok"
        keterangan="Panjang batang = median harga. Sorot batang untuk melihat kuartil, rata-rata, dan jumlah data."
        aksi={
          <Pilihan
            label="Kelompokkan menurut"
            nilai={kelompok as string}
            opsi={PENGELOMPOKAN.map((p) => ({ nilai: p.kunci as string, label: p.label }))}
            onUbah={(v) => setKelompok(v as keyof Eda)}
          />
        }
      >
        <GrafikBatangH
          lebarLabel={150}
          formatNilai={(v) => formatEuro(v)}
          data={grup.map((g) => ({ label: g.grup, nilai: g.median, ...g }))}
          tooltip={(d) => {
            const g = grup.find((x) => x.grup === d.label)!;
            return {
              judul: d.label,
              baris: [
                ["Jumlah laptop", g.jumlah],
                ["Median", formatEuro(g.median)],
                ["Kuartil 1 – 3", `${formatEuro(g.q1)} – ${formatEuro(g.q3)}`],
                ["Rata-rata", formatEuro(g.rata_rata)],
                ["Min – maks", `${formatEuro(g.min)} – ${formatEuro(g.max)}`],
              ],
            };
          }}
        />
      </Kartu>

      <Kartu
        judul="Sebaran fitur vs harga"
        keterangan="Setiap titik satu laptop. Sumbu harga memakai skala logaritmik agar laptop murah dan mahal sama-sama terlihat."
        aksi={
          <Pilihan
            label="Sumbu X"
            nilai={sumbuX}
            opsi={SUMBU_SEBAR.map((s) => ({ nilai: s.kunci, label: s.label }))}
            onUbah={setSumbuX}
          />
        }
      >
        <GrafikSebar
          titik={e.sebaran.map((s) => ({
            x: s[sumbuX],
            y: s.harga,
            judul: `${s.merek} · ${s.tipe}`,
          }))}
          labelX={sumbu.label}
          labelY="Harga (euro)"
          formatX={(v) => formatAngka(v, sumbuX === "berat" ? 1 : 0)}
          formatY={(v) => formatAngka(v, 0)}
          logX={sumbu.log}
          logXPangkatDua
          logY
        />
      </Kartu>

      <Kartu judul="Statistik deskriptif fitur numerik" keterangan="Dihitung dari data yang dipakai (setelah parsing teks menjadi angka).">
        <div className="overflow-x-auto">
          <table className="tabel">
            <thead>
              <tr>
                <th>Fitur</th>
                <th className="kanan">Rata-rata</th>
                <th className="kanan">Std</th>
                <th className="kanan">Min</th>
                <th className="kanan">Q1</th>
                <th className="kanan">Median</th>
                <th className="kanan">Q3</th>
                <th className="kanan">Maks</th>
                <th className="kanan">Skewness</th>
              </tr>
            </thead>
            <tbody>
              {r.statistik_numerik.map((s) => (
                <tr key={s.kolom}>
                  <td>{s.label}</td>
                  {[s.rata_rata, s.std, s.min, s.q1, s.median, s.q3, s.max, s.skewness].map((v, i) => (
                    <td key={i} className="kanan">
                      {formatAngka(v, 2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Kartu>

      <Kartu judul="Contoh data mentah" keterangan={`5 baris pertama ${r.nama_file}, apa adanya sebelum diproses.`}>
        <div className="overflow-x-auto">
          <table className="tabel text-xs">
            <thead>
              <tr>
                {r.kolom_mentah.map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {r.contoh_baris.map((b, i) => (
                <tr key={i}>
                  {r.kolom_mentah.map((k) => (
                    <td key={k} className="whitespace-nowrap">
                      {String(b[k])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Kartu>
    </div>
  );
}
