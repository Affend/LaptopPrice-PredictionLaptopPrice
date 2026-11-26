/** format.ts — Format angka gaya Indonesia (titik ribuan, koma desimal). */

const id = (opsi: Intl.NumberFormatOptions) => new Intl.NumberFormat("id-ID", opsi);

export function formatRupiah(n: number): string {
  return id({ style: "currency", currency: "IDR", maximumFractionDigits: 0 }).format(n);
}

/** Rp26,9 juta / Rp1,2 miliar — untuk angka besar yang perlu dibaca cepat. */
export function formatRupiahSingkat(n: number): string {
  if (Math.abs(n) >= 1e9) return `Rp${formatAngka(n / 1e9, 2)} miliar`;
  if (Math.abs(n) >= 1e6) return `Rp${formatAngka(n / 1e6, 1)} juta`;
  return formatRupiah(n);
}

export function formatEuro(n: number, desimal = 0): string {
  return id({ style: "currency", currency: "EUR", maximumFractionDigits: desimal, minimumFractionDigits: desimal }).format(n);
}

export function formatAngka(n: number, desimal = 2): string {
  return id({ maximumFractionDigits: desimal, minimumFractionDigits: desimal }).format(n);
}

export function formatPersen(n: number, desimal = 1, tanda = false): string {
  const teks = `${formatAngka(n, desimal)}%`;
  return tanda && n > 0 ? `+${teks}` : teks;
}

/** Angka sangat kecil/besar dalam notasi ilmiah: 1,8 × 10⁻¹³ */
export function formatIlmiah(n: number, desimal = 1): string {
  if (n === 0) return "0";
  const eksponen = Math.floor(Math.log10(Math.abs(n)));
  if (eksponen >= -3 && eksponen < 6) return formatAngka(n, Math.max(0, 3 - eksponen));
  const mantisa = n / 10 ** eksponen;
  const pangkat = String(eksponen)
    .split("")
    .map((c) => "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"["-0123456789".indexOf(c)])
    .join("");
  return `${formatAngka(mantisa, desimal)} × 10${pangkat}`;
}

export function formatTanggal(iso: string | null | undefined): string {
  if (!iso) return "-";
  const tanggal = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return tanggal.toLocaleString("id-ID", { dateStyle: "medium", timeStyle: "short" });
}
