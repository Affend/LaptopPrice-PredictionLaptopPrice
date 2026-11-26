import Link from "next/link";
import RingkasanBeranda from "@/components/RingkasanBeranda";
import { Kartu } from "@/components/ui";

const ALUR = [
  { judul: "Dataset CSV", isi: "Kaggle “Laptop Price”: spesifikasi & harga laptop dalam euro" },
  { judul: "Preprocessing", isi: "Parsing teks → angka, PPI, log2 kapasitas, one-hot, standardisasi" },
  { judul: "Training & validasi", isi: "Split 80/20, 5-Fold CV, 7 model dibandingkan dengan baseline" },
  { judul: "Penyimpanan", isi: "Model → file .joblib · Hasil evaluasi → PostgreSQL (per versi)" },
  { judul: "API FastAPI", isi: "Memuat model aktif, melayani prediksi & data evaluasi" },
  { judul: "Website Next.js", isi: "Grafik interaktif, form prediksi, halaman validasi" },
];

const HALAMAN = [
  { href: "/eksplorasi", judul: "Eksplorasi Data", isi: "Statistik dataset, distribusi harga, korelasi, harga per merek." },
  { href: "/prediksi", judul: "Prediksi Harga", isi: "Isi spesifikasi, lihat perkiraan harga, rentang, dan kontribusi fitur." },
  { href: "/evaluasi", judul: "Validasi & Evaluasi", isi: "Perbandingan model vs baseline, CV, residual, verifikasi manual vs sklearn." },
  { href: "/metode", judul: "Penjelasan Metode", isi: "Rumus regresi linear, normal equation, gradient descent, dan metrik." },
];

export default function Beranda() {
  return (
    <div className="space-y-8">
      <section className="pt-2">
        <p className="text-sm font-medium text-biru">Tugas Besar Kecerdasan Buatan · Pengenalan Pola · Regresi</p>
        <h1 className="mt-2 max-w-3xl text-3xl font-semibold tracking-tight sm:text-4xl">
          LaptopPrice: prediksi harga laptop dari spesifikasinya
        </h1>
        <p className="mt-3 max-w-3xl text-lg text-tinta-2">
          Sistem ini memperkirakan harga laptop berdasarkan merek, prosesor, RAM, penyimpanan, layar, dan fitur lain
          menggunakan <b className="text-tinta">Regresi Linear Berganda</b> yang diimplementasikan manual dengan NumPy, lalu
          memvalidasi hasilnya secara menyeluruh.
        </p>
        <div className="mt-5 flex flex-wrap gap-3">
          <Link href="/prediksi" className="rounded-lg bg-biru px-4 py-2 font-medium text-white hover:bg-biru-tua">
            Coba prediksi harga
          </Link>
          <Link href="/evaluasi" className="rounded-lg border border-sumbu bg-permukaan px-4 py-2 font-medium hover:bg-halaman">
            Lihat hasil validasi
          </Link>
        </div>
      </section>

      <RingkasanBeranda />

      <div className="grid gap-6 lg:grid-cols-2">
        <Kartu judul="Latar belakang">
          <div className="space-y-3 text-tinta-2">
            <p>
              Harga laptop sangat beragam dan dipengaruhi banyak spesifikasi sekaligus. Bagi pembeli, sulit menilai apakah
              harga sebuah laptop wajar; bagi penjual, sulit menentukan harga yang bersaing untuk konfigurasi tertentu.
            </p>
            <p>
              Hubungan spesifikasi dengan harga dapat dipelajari dari data. Regresi linear berganda dipilih karena
              koefisiennya bisa ditafsirkan langsung: berapa persen harga naik jika RAM digandakan, atau berapa selisih harga
              antar-merek dengan spesifikasi yang sama.
            </p>
          </div>
        </Kartu>
        <Kartu judul="Tujuan">
          <ol className="list-decimal space-y-2 pl-5 text-tinta-2">
            <li>Membangun model regresi linear berganda (Normal Equation & Gradient Descent) secara manual dengan NumPy.</li>
            <li>Membandingkannya dengan scikit-learn, Ridge, regresi polinomial, dan baseline rata-rata.</li>
            <li>
              Memvalidasi model secara jujur: data uji terpisah, 5-Fold Cross Validation, analisis residual, dan verifikasi
              perhitungan manual terhadap library.
            </li>
            <li>Menyajikan prediksi beserta rentang ketidakpastian dan penjelasan kontribusi tiap fitur.</li>
          </ol>
        </Kartu>
      </div>

      <Kartu judul="Diagram alur sistem" keterangan="Satu sumber kebenaran: semua angka di website berasal dari hasil training yang tersimpan di database.">
        <ol className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
          {ALUR.map((langkah, i) => (
            <li key={langkah.judul} className="relative rounded-lg border border-garis bg-halaman p-3">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-biru text-xs font-semibold text-white">
                  {i + 1}
                </span>
                <span className="text-sm font-semibold">{langkah.judul}</span>
              </div>
              <p className="mt-2 text-xs text-tinta-2">{langkah.isi}</p>
              {i < ALUR.length - 1 && (
                <span aria-hidden className="absolute -right-2.5 top-1/2 z-10 hidden -translate-y-1/2 text-redup lg:block">
                  ▶
                </span>
              )}
            </li>
          ))}
        </ol>
      </Kartu>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {HALAMAN.map((h) => (
          <Link key={h.href} href={h.href} className="group rounded-xl border border-garis bg-permukaan p-4 hover:border-biru">
            <div className="font-semibold group-hover:text-biru">{h.judul} →</div>
            <p className="mt-1 text-sm text-tinta-2">{h.isi}</p>
          </Link>
        ))}
      </div>
    </div>
  );
}
