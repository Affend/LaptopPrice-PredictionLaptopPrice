"use client";

/**
 * api.ts — Cara frontend mengambil data dari backend FastAPI.
 *
 * Semua angka yang tampil di website berasal dari sini (tidak ada angka hasil
 * yang ditulis manual di halaman). Alamat backend diatur lewat variabel
 * NEXT_PUBLIC_API_URL (default http://localhost:8000).
 */

import { useCallback, useEffect, useState } from "react";

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

/** Galat dari API, membawa kode status HTTP (0 = backend tidak bisa dihubungi). */
export class GalatApi extends Error {
  constructor(
    message: string,
    public status: number,
    public rincian: { kolom: string; pesan: string }[] = [],
  ) {
    super(message);
  }
}

async function bacaGalat(respons: Response): Promise<GalatApi> {
  let badan: unknown = null;
  try {
    badan = await respons.json();
  } catch {
    /* badan bukan JSON */
  }
  const detail = (badan as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return new GalatApi(detail, respons.status);
  if (detail && typeof detail === "object" && "pesan" in detail) {
    const d = detail as { pesan: string; kesalahan?: { kolom: string; pesan: string }[] };
    return new GalatApi(d.pesan, respons.status, d.kesalahan ?? []);
  }
  return new GalatApi(`Backend menjawab dengan status ${respons.status}.`, respons.status);
}

async function kirim<T>(path: string, init?: RequestInit): Promise<T> {
  let respons: Response;
  try {
    respons = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new GalatApi(`Backend tidak dapat dihubungi di ${API_URL}.`, 0);
  }
  if (!respons.ok) throw await bacaGalat(respons);
  return (await respons.json()) as T;
}

export function ambilJson<T>(path: string): Promise<T> {
  return kirim<T>(path);
}

export function kirimJson<T>(path: string, data: unknown): Promise<T> {
  return kirim<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

/**
 * Hook React untuk mengambil data GET.
 * - `memuat` true selama permintaan untuk `path` saat ini belum selesai.
 * - `galat` berisi pesan jika backend mati / menjawab galat.
 * - `muatUlang()` mengulang permintaan (tombol "Coba lagi").
 * path = null berarti belum perlu mengambil apa pun.
 */
export function useApi<T>(path: string | null) {
  const [hasil, setHasil] = useState<{ kunci: string; data?: T; galat?: GalatApi } | null>(null);
  const [percobaan, setPercobaan] = useState(0);
  const kunci = path === null ? null : `${path}#${percobaan}`;

  useEffect(() => {
    if (path === null || kunci === null) return;
    let dibatalkan = false;
    ambilJson<T>(path)
      .then((data) => !dibatalkan && setHasil({ kunci, data }))
      .catch((e: unknown) => {
        if (dibatalkan) return;
        const galat = e instanceof GalatApi ? e : new GalatApi(String(e), -1);
        setHasil({ kunci, galat });
      });
    return () => {
      dibatalkan = true;
    };
  }, [path, kunci]);

  const muatUlang = useCallback(() => setPercobaan((n) => n + 1), []);
  const selesai = hasil !== null && hasil.kunci === kunci;
  return {
    data: selesai ? hasil.data : undefined,
    galat: selesai ? hasil.galat : undefined,
    memuat: path !== null && !selesai,
    muatUlang,
    /** hasil permintaan terakhir yang sudah selesai (tetap ada selama memuat ulang) */
    terakhir: hasil,
  };
}
