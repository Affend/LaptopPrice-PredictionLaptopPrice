/**
 * ui.tsx — Komponen tampilan dasar yang dipakai di semua halaman.
 */

import type { ReactNode } from "react";
import { API_URL, GalatApi } from "@/lib/api";

export function JudulHalaman({ judul, keterangan }: { judul: string; keterangan?: ReactNode }) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">{judul}</h1>
      {keterangan && <p className="mt-2 max-w-3xl text-tinta-2">{keterangan}</p>}
    </header>
  );
}

export function Kartu({
  judul,
  keterangan,
  children,
  id,
  aksi,
}: {
  judul?: string;
  keterangan?: ReactNode;
  children: ReactNode;
  id?: string;
  aksi?: ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-20 rounded-xl border border-garis bg-permukaan p-4 sm:p-6">
      {(judul || aksi) && (
        <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {judul && <h2 className="text-lg font-semibold">{judul}</h2>}
            {keterangan && <div className="mt-1 max-w-3xl text-sm text-tinta-2">{keterangan}</div>}
          </div>
          {aksi}
        </div>
      )}
      {children}
    </section>
  );
}

export function StatTile({ label, nilai, catatan }: { label: string; nilai: ReactNode; catatan?: ReactNode }) {
  return (
    <div className="rounded-xl border border-garis bg-permukaan p-4">
      <div className="text-sm text-tinta-2">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{nilai}</div>
      {catatan && <div className="mt-1 text-xs text-redup">{catatan}</div>}
    </div>
  );
}

export function Memuat({ teks = "Memuat data dari backend…" }: { teks?: string }) {
  return (
    <div role="status" className="flex items-center gap-3 rounded-xl border border-garis bg-permukaan p-6 text-tinta-2">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-sumbu border-t-biru" aria-hidden />
      {teks}
    </div>
  );
}

/** Pesan galat yang menjelaskan apa yang harus dilakukan pengguna. */
export function PesanGalat({ galat, onCobaLagi }: { galat: GalatApi; onCobaLagi?: () => void }) {
  const backendMati = galat.status === 0;
  return (
    <div role="alert" className="rounded-xl border border-merah/40 bg-red-50 p-5">
      <div className="flex items-start gap-3">
        <span aria-hidden className="mt-0.5 text-merah">⛔</span>
        <div className="space-y-2 text-sm">
          <p className="font-semibold text-tinta">{backendMati ? "Backend tidak dapat dihubungi" : galat.message}</p>
          {backendMati ? (
            <>
              <p className="text-tinta-2">
                Website ini mengambil semua data dari API di <code className="rounded bg-white px-1">{API_URL}</code>.
                Pastikan backend sudah dijalankan:
              </p>
              <pre className="overflow-x-auto rounded bg-white p-2 text-xs">
                cd backend{"\n"}.venv\Scripts\activate{"\n"}uvicorn app.main:app --port 8000
              </pre>
              <p className="text-tinta-2">Atau dengan Docker: <code className="rounded bg-white px-1">docker compose up</code></p>
            </>
          ) : (
            galat.status > 0 && <p className="text-tinta-2">Kode status HTTP: {galat.status}</p>
          )}
          {galat.rincian.length > 0 && (
            <ul className="list-disc pl-5 text-tinta-2">
              {galat.rincian.map((r, i) => (
                <li key={i}>
                  <b>{r.kolom}</b>: {r.pesan}
                </li>
              ))}
            </ul>
          )}
          {onCobaLagi && (
            <button
              onClick={onCobaLagi}
              className="rounded-lg border border-sumbu bg-white px-3 py-1.5 font-medium hover:bg-halaman"
            >
              Coba lagi
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/** Menampilkan status memuat / galat. Mengembalikan null jika data sudah ada. */
export function StatusApi({
  memuat,
  galat,
  muatUlang,
  teks,
}: {
  memuat: boolean;
  galat?: GalatApi;
  muatUlang?: () => void;
  teks?: string;
}) {
  if (galat) return <PesanGalat galat={galat} onCobaLagi={muatUlang} />;
  if (memuat) return <Memuat teks={teks} />;
  return null;
}

export function KotakInfo({
  jenis = "info",
  judul,
  children,
}: {
  jenis?: "info" | "peringatan";
  judul?: string;
  children: ReactNode;
}) {
  const gaya =
    jenis === "peringatan" ? "border-peringatan bg-peringatan-muda" : "border-biru/30 bg-biru-muda";
  return (
    <div className={`rounded-lg border p-4 text-sm ${gaya}`}>
      <div className="flex gap-2">
        <span aria-hidden>{jenis === "peringatan" ? "⚠️" : "ℹ️"}</span>
        <div className="space-y-1">
          {judul && <p className="font-semibold">{judul}</p>}
          <div className="text-tinta-2">{children}</div>
        </div>
      </div>
    </div>
  );
}

export function Lencana({ children, warna = "abu" }: { children: ReactNode; warna?: "abu" | "biru" | "hijau" }) {
  const gaya = {
    abu: "bg-halaman text-tinta-2 border-garis",
    biru: "bg-biru-muda text-biru-tua border-biru/30",
    hijau: "bg-green-50 text-baik border-green-200",
  }[warna];
  return <span className={`inline-block rounded-full border px-2 py-0.5 text-xs font-medium ${gaya}`}>{children}</span>;
}
