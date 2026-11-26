/**
 * Rumus.tsx — Menampilkan rumus LaTeX dengan KaTeX.
 * Dirender di server (tanpa JavaScript di browser), jadi halaman tetap ringan.
 */

import katex from "katex";

export default function Rumus({ tex, blok = false }: { tex: string; blok?: boolean }) {
  const html = katex.renderToString(tex, { displayMode: blok, throwOnError: false, strict: false });
  return blok ? (
    <div className="my-3" dangerouslySetInnerHTML={{ __html: html }} />
  ) : (
    <span dangerouslySetInnerHTML={{ __html: html }} />
  );
}
