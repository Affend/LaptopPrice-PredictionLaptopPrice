"use client";

/**
 * grafik.tsx — Pembungkus grafik Recharts dengan gaya yang seragam.
 *
 * Aturan gaya: garis grid & sumbu tipis berwarna redup, warna seri mengikuti
 * palet tervalidasi (lib/warna.ts), setiap grafik punya tooltip saat disorot,
 * dan grafik dengan >= 2 seri selalu punya legenda.
 */

import type { ReactNode } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { formatAngka } from "@/lib/format";
import { WARNA, tintaDiAtas, warnaKorelasi } from "@/lib/warna";

const TICK = { fill: WARNA.tintaRedup, fontSize: 12 };
const SUMBU = { stroke: WARNA.sumbu };

/**
 * Tick untuk sumbu logaritmik: deret 1-2-5 (atau pangkat 10 saja jika
 * rentangnya sangat lebar). Ditentukan sendiri karena pembuat tick bawaan
 * Recharts bisa menghasilkan tick ganda pada skala log.
 */
export function ticksLog(nilai: number[], pangkatDua = false): { ticks: number[]; domain: [number, number] } {
  const positif = nilai.filter((v) => v > 0 && Number.isFinite(v));
  if (!positif.length) return { ticks: [], domain: [1, 10] };
  const min = Math.min(...positif);
  const max = Math.max(...positif);
  let ticks: number[] = [];
  if (pangkatDua) {
    for (let e = Math.floor(Math.log2(min)); e <= Math.ceil(Math.log2(max)); e++) ticks.push(2 ** e);
  } else {
    const e0 = Math.floor(Math.log10(min));
    const e1 = Math.ceil(Math.log10(max));
    const pengali = e1 - e0 > 4 ? [1] : [1, 2, 5];
    for (let e = e0; e <= e1; e++) for (const m of pengali) ticks.push(m * 10 ** e);
  }
  ticks = ticks.filter((v) => v >= min * 0.999 && v <= max * 1.001);
  if (ticks.length < 2) ticks = [min, max];
  return { ticks: Array.from(new Set(ticks)), domain: [min, max] };
}

// ---------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------
export function KotakTooltip({ judul, baris }: { judul?: ReactNode; baris: [ReactNode, ReactNode][] }) {
  return (
    <div className="rounded-lg border border-garis bg-white px-3 py-2 text-xs shadow-sm">
      {judul && <div className="mb-1 font-semibold text-tinta">{judul}</div>}
      <table>
        <tbody>
          {baris.map(([k, v], i) => (
            <tr key={i}>
              <td className="pr-3 text-tinta-2">{k}</td>
              <td className="angka text-right font-medium text-tinta">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

type IsiTooltip<T> = (d: T) => { judul?: ReactNode; baris: [ReactNode, ReactNode][] };

/** Fungsi render untuk prop `content` Tooltip Recharts (bukan komponen, agar tidak dibuat ulang tiap render). */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function tampilkanTooltip<T>(isi: IsiTooltip<T>, props: { active?: boolean; payload?: readonly any[] }) {
  const { active, payload } = props;
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as T;
  const { judul, baris } = isi(d);
  return <KotakTooltip judul={judul} baris={baris} />;
}

// ---------------------------------------------------------------------------
// Histogram (opsional + kurva normal pembanding)
// ---------------------------------------------------------------------------
export function GrafikHistogram({
  data,
  labelX,
  formatX = (v) => formatAngka(v, 2),
  warna = WARNA.seri1,
  garisNormal = false,
  tinggi = 260,
}: {
  data: { tengah: number; bin_kiri: number; bin_kanan: number; frekuensi: number; frekuensi_normal?: number }[];
  labelX: string;
  formatX?: (v: number) => string;
  warna?: string;
  garisNormal?: boolean;
  tinggi?: number;
}) {
  const isi: IsiTooltip<(typeof data)[number]> = (d) => ({
    judul: `${formatX(d.bin_kiri)} – ${formatX(d.bin_kanan)}`,
    baris: [
      ["Jumlah", d.frekuensi],
      ...(garisNormal && d.frekuensi_normal !== undefined
        ? ([["Harapan jika normal", formatAngka(d.frekuensi_normal, 1)]] as [ReactNode, ReactNode][])
        : []),
    ],
  });
  return (
    <ResponsiveContainer width="100%" height={tinggi}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, bottom: 24, left: 0 }} barCategoryGap={1}>
        <CartesianGrid stroke={WARNA.grid} vertical={false} />
        <XAxis
          dataKey="tengah"
          tick={TICK}
          axisLine={SUMBU}
          tickLine={false}
          tickFormatter={(v) => formatX(Number(v))}
          label={{ value: labelX, position: "insideBottom", offset: -14, fill: WARNA.tintaKedua, fontSize: 12 }}
          minTickGap={24}
        />
        <YAxis tick={TICK} axisLine={false} tickLine={false} width={40} />
        <Tooltip content={(p) => tampilkanTooltip(isi, p)} cursor={{ fill: "rgba(11,11,11,0.04)" }} />
        <Bar dataKey="frekuensi" name="Jumlah" fill={warna} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        {garisNormal && (
          <Line
            dataKey="frekuensi_normal"
            name="Kurva normal"
            stroke={WARNA.seri2}
            strokeWidth={2}
            dot={false}
            type="monotone"
            isAnimationActive={false}
          />
        )}
        {garisNormal && <Legend verticalAlign="top" height={28} iconType="plainline" wrapperStyle={{ fontSize: 12 }} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Batang horizontal (opsional divergen: positif biru, negatif merah)
// ---------------------------------------------------------------------------
export function GrafikBatangH({
  data,
  formatNilai = (v) => formatAngka(v, 2),
  divergen = false,
  warna = WARNA.seri1,
  lebarLabel = 140,
  tooltip,
  garisNol = divergen,
}: {
  data: { label: string; nilai: number; info?: [ReactNode, ReactNode][] }[];
  formatNilai?: (v: number) => string;
  divergen?: boolean;
  warna?: string;
  lebarLabel?: number;
  tooltip?: IsiTooltip<{ label: string; nilai: number; info?: [ReactNode, ReactNode][] }>;
  garisNol?: boolean;
}) {
  const isi: IsiTooltip<(typeof data)[number]> =
    tooltip ?? ((d) => ({ judul: d.label, baris: [["Nilai", formatNilai(d.nilai)], ...(d.info ?? [])] }));
  const tinggi = Math.max(120, data.length * 30 + 40);
  return (
    <ResponsiveContainer width="100%" height={tinggi}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 4 }} barCategoryGap={4}>
        <CartesianGrid stroke={WARNA.grid} horizontal={false} />
        <XAxis type="number" tick={TICK} axisLine={SUMBU} tickLine={false} tickFormatter={(v) => formatNilai(Number(v))} />
        <YAxis type="category" dataKey="label" tick={{ ...TICK, fill: WARNA.tintaKedua }} width={lebarLabel} axisLine={false} tickLine={false} interval={0} />
        <Tooltip content={(p) => tampilkanTooltip(isi, p)} cursor={{ fill: "rgba(11,11,11,0.04)" }} />
        {garisNol && <ReferenceLine x={0} stroke={WARNA.tintaRedup} />}
        <Bar dataKey="nilai" radius={3} isAnimationActive={false}>
          {data.map((d, i) => (
            <Cell key={i} fill={divergen ? (d.nilai >= 0 ? WARNA.positif : WARNA.negatif) : warna} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Batang vertikal berkelompok (mis. RMSE CV vs RMSE uji per model)
// ---------------------------------------------------------------------------
export function GrafikBatangKelompok({
  data,
  kunciLabel,
  seri,
  formatNilai = (v) => formatAngka(v, 0),
  labelY,
  tinggi = 300,
}: {
  data: Record<string, string | number>[];
  kunciLabel: string;
  seri: { key: string; label: string; warna: string }[];
  formatNilai?: (v: number) => string;
  labelY?: string;
  tinggi?: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={tinggi}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 8 }} barGap={2}>
        <CartesianGrid stroke={WARNA.grid} vertical={false} />
        <XAxis dataKey={kunciLabel} tick={{ ...TICK, fontSize: 11 }} axisLine={SUMBU} tickLine={false} interval={0} />
        <YAxis
          tick={TICK}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => formatNilai(Number(v))}
          label={labelY ? { value: labelY, angle: -90, position: "insideLeft", fill: WARNA.tintaKedua, fontSize: 12 } : undefined}
        />
        <Tooltip
          formatter={(v) => formatNilai(Number(v))}
          contentStyle={{ fontSize: 12, borderRadius: 8, borderColor: WARNA.grid }}
          cursor={{ fill: "rgba(11,11,11,0.04)" }}
        />
        <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />
        {seri.map((s) => (
          <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.warna} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Diagram sebar (scatter) dengan garis acuan
// ---------------------------------------------------------------------------
export interface TitikSebar {
  x: number;
  y: number;
  judul?: string;
  info?: [ReactNode, ReactNode][];
}

export function GrafikSebar({
  titik,
  labelX,
  labelY,
  formatX = (v) => formatAngka(v, 0),
  formatY = (v) => formatAngka(v, 0),
  garis,
  logX = false,
  logY = false,
  logXPangkatDua = false,
  tinggi = 340,
  warna = WARNA.seri1,
}: {
  titik: TitikSebar[];
  labelX: string;
  labelY: string;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  /** identitas: y = x mulai dari 0 · diagonal: y = x di seluruh rentang (Q-Q plot) · nol: y = 0 */
  garis?: "identitas" | "diagonal" | "nol";
  logX?: boolean;
  logY?: boolean;
  logXPangkatDua?: boolean;
  tinggi?: number;
  warna?: string;
}) {
  const xs = titik.map((t) => t.x);
  const ys = titik.map((t) => t.y);
  // dibulatkan ke atas ke kelipatan "rapi" (mis. 6.221 -> 6.500) agar tick sumbu enak dibaca
  const maks = Math.max(...xs, ...ys) * 1.02;
  const langkah = maks > 0 ? 10 ** Math.floor(Math.log10(maks)) / 2 : 1;
  const batas = garis === "identitas" ? Math.ceil(maks / langkah) * langkah / 1.05 : Math.max(...xs, ...ys);
  const bawah = Math.min(...xs, ...ys);
  const tickX = logX ? ticksLog(xs, logXPangkatDua) : null;
  const tickY = logY ? ticksLog(ys) : null;
  const isi: IsiTooltip<TitikSebar> = (d) => ({
    judul: d.judul,
    baris: [[labelX, formatX(d.x)], [labelY, formatY(d.y)], ...(d.info ?? [])],
  });
  return (
    <ResponsiveContainer width="100%" height={tinggi}>
      <ScatterChart margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
        <CartesianGrid stroke={WARNA.grid} />
        <XAxis
          type="number"
          dataKey="x"
          name={labelX}
          scale={logX ? "log" : "auto"}
          domain={tickX ? tickX.domain : garis === "identitas" ? [0, batas * 1.05] : ["auto", "auto"]}
          ticks={tickX?.ticks}
          tick={TICK}
          axisLine={SUMBU}
          tickLine={false}
          tickFormatter={(v) => formatX(Number(v))}
          label={{ value: labelX, position: "insideBottom", offset: -16, fill: WARNA.tintaKedua, fontSize: 12 }}
          allowDataOverflow={false}
        />
        <YAxis
          type="number"
          dataKey="y"
          name={labelY}
          scale={logY ? "log" : "auto"}
          domain={tickY ? tickY.domain : garis === "identitas" ? [0, batas * 1.05] : ["auto", "auto"]}
          ticks={tickY?.ticks}
          tick={TICK}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => formatY(Number(v))}
          width={64}
          label={{ value: labelY, angle: -90, position: "insideLeft", fill: WARNA.tintaKedua, fontSize: 12, dx: -2 }}
        />
        <ZAxis range={[48, 48]} />
        <Tooltip content={(p) => tampilkanTooltip(isi, p)} cursor={{ strokeDasharray: "3 3", stroke: WARNA.sumbu }} />
        {garis === "identitas" && (
          <ReferenceLine
            segment={[
              { x: Math.min(0, bawah), y: Math.min(0, bawah) },
              { x: batas * 1.05, y: batas * 1.05 },
            ]}
            stroke={WARNA.tintaRedup}
            strokeDasharray="5 4"
            label={{ value: "prediksi = aktual", position: "insideTopLeft", fill: WARNA.tintaKedua, fontSize: 11 }}
          />
        )}
        {garis === "diagonal" && (
          <ReferenceLine
            segment={[
              { x: bawah, y: bawah },
              { x: batas, y: batas },
            ]}
            stroke={WARNA.tintaRedup}
            strokeDasharray="5 4"
            ifOverflow="extendDomain"
          />
        )}
        {garis === "nol" && <ReferenceLine y={0} stroke={WARNA.tintaRedup} strokeDasharray="5 4" />}
        <Scatter data={titik} fill={warna} fillOpacity={0.6} stroke={WARNA.permukaan} strokeWidth={1} isAnimationActive={false} />
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Grafik garis (learning curve, kurva loss, jalur koefisien, pemilihan alpha)
// ---------------------------------------------------------------------------
export function GrafikGaris({
  data,
  kunciX,
  seri,
  labelX,
  labelY,
  formatX = (v) => formatAngka(v, 0),
  formatY = (v) => formatAngka(v, 2),
  logX = false,
  logY = false,
  garisAcuan,
  tinggi = 300,
  xKategori = false,
}: {
  data: Record<string, number | string | null>[];
  kunciX: string;
  seri: { key: string; label: string; warna: string; putus?: boolean; titik?: boolean }[];
  labelX: string;
  labelY: string;
  formatX?: (v: number) => string;
  formatY?: (v: number) => string;
  logX?: boolean;
  logY?: boolean;
  garisAcuan?: { y?: number; x?: number; label: string }[];
  tinggi?: number;
  /** true: sumbu X berupa kategori (label teks), bukan angka */
  xKategori?: boolean;
}) {
  const tickX = logX ? ticksLog(data.map((d) => Number(d[kunciX]))) : null;
  const semuaY = data.flatMap((d) => seri.map((s) => d[s.key])).filter((v): v is number => typeof v === "number");
  const tickY = logY ? ticksLog([...semuaY, ...(garisAcuan?.map((g) => g.y ?? NaN) ?? [])]) : null;
  return (
    <ResponsiveContainer width="100%" height={tinggi}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 28, left: 8 }}>
        <CartesianGrid stroke={WARNA.grid} vertical={false} />
        {xKategori ? (
          <XAxis
            dataKey={kunciX}
            type="category"
            tick={TICK}
            axisLine={SUMBU}
            tickLine={false}
            label={{ value: labelX, position: "insideBottom", offset: -16, fill: WARNA.tintaKedua, fontSize: 12 }}
          />
        ) : (
          <XAxis
            dataKey={kunciX}
            type="number"
            scale={logX ? "log" : "auto"}
            domain={tickX ? tickX.domain : ["auto", "auto"]}
            ticks={tickX?.ticks}
            tick={TICK}
            axisLine={SUMBU}
            tickLine={false}
            tickFormatter={(v) => formatX(Number(v))}
            label={{ value: labelX, position: "insideBottom", offset: -16, fill: WARNA.tintaKedua, fontSize: 12 }}
          />
        )}
        <YAxis
          scale={logY ? "log" : "auto"}
          domain={tickY ? tickY.domain : ["auto", "auto"]}
          ticks={tickY?.ticks}
          tick={TICK}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v) => formatY(Number(v))}
          width={64}
          label={{ value: labelY, angle: -90, position: "insideLeft", fill: WARNA.tintaKedua, fontSize: 12, dx: -2 }}
        />
        <Tooltip
          labelFormatter={(v) => `${labelX}: ${xKategori ? String(v) : formatX(Number(v))}`}
          formatter={(v, nama) => [v === null || v === undefined ? "-" : formatY(Number(v)), nama]}
          contentStyle={{ fontSize: 12, borderRadius: 8, borderColor: WARNA.grid }}
        />
        {seri.length > 1 && <Legend verticalAlign="top" height={30} wrapperStyle={{ fontSize: 12 }} />}
        {garisAcuan?.map((g, i) => (
          <ReferenceLine
            key={i}
            y={g.y}
            x={g.x}
            stroke={WARNA.tintaRedup}
            strokeDasharray="5 4"
            label={{ value: g.label, position: "insideTopRight", fill: WARNA.tintaKedua, fontSize: 11 }}
          />
        ))}
        {seri.map((s) => (
          <Line
            key={s.key}
            dataKey={s.key}
            name={s.label}
            stroke={s.warna}
            strokeWidth={2}
            strokeDasharray={s.putus ? "6 4" : undefined}
            dot={s.titik ? { r: 4, fill: s.warna, stroke: WARNA.permukaan, strokeWidth: 2 } : false}
            activeDot={{ r: 5 }}
            connectNulls
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Heatmap korelasi (HTML biasa: sel berwarna + angka, tooltip lewat title)
// ---------------------------------------------------------------------------
export function HeatmapKorelasi({ kolom, nilai, label }: { kolom: string[]; nilai: number[][]; label?: Record<string, string> }) {
  const nama = (k: string) => label?.[k] ?? k;
  return (
    <div className="overflow-x-auto">
      <table className="border-separate" style={{ borderSpacing: 2 }}>
        <thead>
          <tr>
            <th />
            {kolom.map((k) => (
              <th key={k} className="h-28 w-12 align-bottom text-xs font-normal text-tinta-2">
                <div className="mx-auto w-4 rotate-180 whitespace-nowrap [writing-mode:vertical-rl]">{nama(k)}</div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {kolom.map((baris, i) => (
            <tr key={baris}>
              <th className="whitespace-nowrap pr-2 text-right text-xs font-normal text-tinta-2">{nama(baris)}</th>
              {kolom.map((k, j) => {
                const r = nilai[i][j];
                return (
                  <td
                    key={k}
                    title={`${nama(baris)} × ${nama(k)}: r = ${formatAngka(r, 3)}`}
                    className="angka h-10 w-12 rounded text-center text-[11px]"
                    style={{ background: warnaKorelasi(r), color: tintaDiAtas(r) }}
                  >
                    {formatAngka(r, 2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-3 flex items-center gap-2 text-xs text-tinta-2">
        <span>−1</span>
        <span className="h-2 w-40 rounded" style={{ background: `linear-gradient(to right, ${warnaKorelasi(-1)}, ${warnaKorelasi(0)}, ${warnaKorelasi(1)})` }} />
        <span>+1</span>
        <span className="ml-2">(merah = berkorelasi negatif, biru = positif)</span>
      </div>
    </div>
  );
}
