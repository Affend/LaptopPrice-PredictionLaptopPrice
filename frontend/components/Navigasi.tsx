"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect } from "react";
import { useApi } from "@/lib/api";
import type { Health } from "@/lib/tipe";

const MENU = [
  { href: "/", label: "Beranda" },
  { href: "/eksplorasi", label: "Eksplorasi Data" },
  { href: "/prediksi", label: "Prediksi Harga" },
  { href: "/evaluasi", label: "Validasi & Evaluasi" },
  { href: "/metode", label: "Penjelasan Metode" },
];

/** Indikator kecil: apakah backend & model aktif siap. Diperiksa ulang tiap 10 detik. */
function StatusBackend() {
  const { terakhir, memuat, muatUlang } = useApi<Health>("/api/health");
  useEffect(() => {
    const id = setInterval(muatUlang, 10_000);
    return () => clearInterval(id);
  }, [muatUlang]);
  // pakai hasil pemeriksaan terakhir agar indikator tidak berkedip saat memeriksa ulang
  const data = terakhir?.data;
  const galat = terakhir?.galat;
  let teks = "Memeriksa backend…";
  let warna = "bg-sumbu";
  if (galat) {
    teks = "Backend tidak terhubung";
    warna = "bg-merah";
  } else if (data) {
    teks = data.status === "ok" ? `Model aktif v${data.model_aktif?.nomor_versi}` : "Backend tanpa model aktif";
    warna = data.status === "ok" ? "bg-baik" : "bg-peringatan";
  }
  return (
    <span className="flex items-center gap-1.5 text-xs text-tinta-2" aria-live="polite" title={memuat ? "" : teks}>
      <span className={`h-2 w-2 rounded-full ${warna}`} aria-hidden />
      {teks}
    </span>
  );
}

export default function Navigasi() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-20 border-b border-garis bg-permukaan/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="font-semibold tracking-tight">
          Laptop<span className="text-biru">Price</span>
        </Link>
        <nav className="-mx-1 flex flex-1 gap-1 overflow-x-auto" aria-label="Menu utama">
          {MENU.map((m) => {
            const aktif = m.href === "/" ? pathname === "/" : pathname.startsWith(m.href);
            return (
              <Link
                key={m.href}
                href={m.href}
                aria-current={aktif ? "page" : undefined}
                className={`whitespace-nowrap rounded-lg px-3 py-1.5 text-sm ${
                  aktif ? "bg-biru-muda font-medium text-biru-tua" : "text-tinta-2 hover:bg-halaman"
                }`}
              >
                {m.label}
              </Link>
            );
          })}
        </nav>
        <StatusBackend />
      </div>
    </header>
  );
}
