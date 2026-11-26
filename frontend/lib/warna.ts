/**
 * warna.ts — Palet grafik.
 *
 * Diambil dari palet referensi yang sudah divalidasi aman untuk buta warna
 * (urutan slot kategorikal tidak boleh diubah). Warna dipakai sesuai FUNGSI:
 * - kategorikal : membedakan seri (urutan tetap, tidak diputar ulang)
 * - divergen    : nilai positif vs negatif (biru vs merah, tengah abu-abu)
 * - status      : peringatan / baik, selalu disertai ikon + teks
 */

export const SERI = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

export const WARNA = {
  seri1: SERI[0],
  seri2: SERI[1],
  positif: "#2a78d6", // menaikkan harga
  negatif: "#e34948", // menurunkan harga
  netral: "#f0efec",
  tintaUtama: "#0b0b0b",
  tintaKedua: "#52514e",
  tintaRedup: "#898781",
  grid: "#e1e0d9",
  sumbu: "#c3c2b7",
  permukaan: "#fcfcfb",
  peringatan: "#fab219",
  baik: "#0ca30c",
  kritis: "#d03b3b",
};

/** Warna sel heatmap korelasi: -1 merah … 0 abu-abu … +1 biru. */
export function warnaKorelasi(r: number): string {
  const t = Math.max(-1, Math.min(1, r));
  const [ujung, tengah] = [t >= 0 ? [42, 120, 214] : [227, 73, 72], [240, 239, 236]];
  const a = Math.abs(t);
  const c = tengah.map((v, i) => Math.round(v + (ujung[i] - v) * a));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

/** Teks di atas sel berwarna: putih jika latar gelap. */
export function tintaDiAtas(r: number): string {
  return Math.abs(r) > 0.55 ? "#ffffff" : WARNA.tintaUtama;
}
