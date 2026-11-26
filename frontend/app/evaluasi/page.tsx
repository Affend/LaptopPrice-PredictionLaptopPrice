"use client";

/**
 * Halaman Validasi & Evaluasi — bagian terpenting tugas.
 * Semua angka diambil dari GET /api/models dan GET /api/evaluation/{id};
 * kalimat interpretasi pun dihitung dari angka tersebut (bukan ditulis manual).
 */

import { useState, type ReactNode } from "react";
import { GrafikBatangH, GrafikBatangKelompok, GrafikGaris, GrafikHistogram, GrafikSebar } from "@/components/grafik";
import { JudulHalaman, Kartu, KotakInfo, Lencana, StatTile, StatusApi } from "@/components/ui";
import { useApi } from "@/lib/api";
import { formatAngka, formatEuro, formatIlmiah, formatPersen, formatTanggal } from "@/lib/format";
import type { DaftarModel, Evaluasi, RingkasModel } from "@/lib/tipe";
import { SERI } from "@/lib/warna";

const DAFTAR_ISI = [
  ["ringkasan", "Ringkasan"],
  ["perbandingan", "Model vs baseline"],
  ["pemilihan", "Pemilihan model"],
  ["cv", "Cross validation"],
  ["prediksi-aktual", "Prediksi vs aktual"],
  ["normalitas", "Normalitas residual"],
  ["interval", "Rentang perkiraan"],
  ["learning-curve", "Learning curve"],
  ["gradient-descent", "Gradient descent"],
  ["verifikasi", "Manual vs sklearn"],
  ["error-grup", "Error per merek/tipe"],
  ["top-error", "10 error terbesar"],
  ["koefisien", "Koefisien"],
  ["multikolinearitas", "Multikolinearitas"],
] as const;

function implementasi(nama: string): string {
  if (nama.startsWith("baseline")) return "pembanding";
  return nama.includes("sklearn") ? "scikit-learn" : "manual (NumPy)";
}

function Angka({ children }: { children: ReactNode }) {
  return <span className="angka font-medium text-tinta">{children}</span>;
}

// ---------------------------------------------------------------------------
// A. Ringkasan
// ---------------------------------------------------------------------------
function Ringkasan({ e, baseline }: { e: Evaluasi; baseline?: RingkasModel }) {
  const uji = e.evaluasi.metrik.uji;
  const b = baseline?.metrik_uji;
  const ds = e.ringkasan_run.dataset;
  const banding = (nilai: number, nilaiB: number | undefined, fmt: (v: number) => string, makinKecil = true) =>
    nilaiB === undefined
      ? undefined
      : `baseline ${fmt(nilaiB)} · ${makinKecil ? "lebih kecil" : "lebih besar"} ${formatPersen(
          Math.abs((1 - nilai / nilaiB) * 100),
          0,
        )}`;
  return (
    <Kartu
      id="ringkasan"
      judul="Ringkasan performa pada data uji"
      keterangan={
        <>
          Data uji ({formatAngka(ds.jumlah_uji, 0)} laptop, 20%) dipisahkan sejak awal dan <b>tidak pernah</b> dipakai untuk
          melatih atau memilih model. Angka di bawah adalah perkiraan jujur performa pada laptop yang belum pernah dilihat model.
        </>
      }
    >
      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-5">
        <StatTile label="MAE" nilai={formatEuro(uji.mae)} catatan={banding(uji.mae, b?.mae, (v) => formatEuro(v))} />
        <StatTile label="RMSE" nilai={formatEuro(uji.rmse)} catatan={banding(uji.rmse, b?.rmse, (v) => formatEuro(v))} />
        <StatTile label="MAPE" nilai={formatPersen(uji.mape)} catatan={banding(uji.mape, b?.mape, (v) => formatPersen(v))} />
        <StatTile label="R² (skala euro)" nilai={formatAngka(uji.r2, 3)} catatan={b ? `baseline ${formatAngka(b.r2, 3)}` : undefined} />
        <StatTile label="R² (skala log)" nilai={formatAngka(uji.r2_log, 3)} catatan="besaran yang dioptimalkan regresi" />
      </div>
      <p className="mt-4 text-sm text-tinta-2">
        Artinya: rata-rata prediksi meleset <Angka>{formatEuro(uji.mae)}</Angka> atau <Angka>{formatPersen(uji.mape)}</Angka> dari
        harga sebenarnya. Model menjelaskan <Angka>{formatPersen(uji.r2_log * 100, 1)}</Angka> variasi ln(harga). RMSE jauh di
        atas MAE menandakan ada sebagian kecil laptop yang meleset sangat jauh (lihat 10 error terbesar).
      </p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// B. Perbandingan model vs baseline
// ---------------------------------------------------------------------------
function Perbandingan({ e, idTerpilih }: { e: Evaluasi; idTerpilih: number }) {
  const run = e.perbandingan_run;
  return (
    <Kartu
      id="perbandingan"
      judul="Perbandingan semua model dengan baseline"
      keterangan="CV = rata-rata ± standar deviasi 5-Fold Cross Validation di data latih. Uji = data uji 20%. Perbaikan dihitung terhadap baseline yang selalu menebak rata-rata harga."
    >
      <div className="overflow-x-auto">
        <table className="tabel">
          <thead>
            <tr>
              <th>Model</th>
              <th>Implementasi</th>
              <th className="kanan">RMSE CV (€)</th>
              <th className="kanan">MAPE CV</th>
              <th className="kanan">R² CV</th>
              <th className="kanan">RMSE uji (€)</th>
              <th className="kanan">MAE uji (€)</th>
              <th className="kanan">MAPE uji</th>
              <th className="kanan">R² log uji</th>
              <th className="kanan">RMSE vs baseline</th>
            </tr>
          </thead>
          <tbody>
            {run.map((m) => (
              <tr key={m.id} className={m.id === idTerpilih ? "sorot" : ""}>
                <td className="min-w-56">
                  {m.label} {m.is_active && <Lencana warna="hijau">aktif</Lencana>}
                </td>
                <td className="text-tinta-2">{implementasi(m.nama_model)}</td>
                <td className="kanan">
                  {formatAngka(m.cv.rmse.rata_rata, 1)} <span className="text-redup">± {formatAngka(m.cv.rmse.std, 1)}</span>
                </td>
                <td className="kanan">
                  {formatPersen(m.cv.mape.rata_rata)} <span className="text-redup">± {formatAngka(m.cv.mape.std, 1)}</span>
                </td>
                <td className="kanan">{formatAngka(m.cv.r2.rata_rata, 3)}</td>
                <td className="kanan">{formatAngka(m.metrik_uji.rmse, 1)}</td>
                <td className="kanan">{formatAngka(m.metrik_uji.mae, 1)}</td>
                <td className="kanan">{formatPersen(m.metrik_uji.mape)}</td>
                <td className="kanan">{formatAngka(m.metrik_uji.r2_log, 3)}</td>
                <td className="kanan">
                  {m.nama_model.startsWith("baseline") ? "—" : formatPersen(-m.perbaikan_vs_baseline.rmse.uji_persen, 1, true)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-6">
        <GrafikBatangKelompok
          data={run.map((m) => ({
            model: m.nama_model.replace("_", " ").replace("_", " "),
            cv: m.cv.rmse.rata_rata,
            uji: m.metrik_uji.rmse,
          }))}
          kunciLabel="model"
          labelY="RMSE (euro)"
          seri={[
            { key: "cv", label: "RMSE 5-Fold CV (data latih)", warna: SERI[0] },
            { key: "uji", label: "RMSE data uji", warna: SERI[1] },
          ]}
        />
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// C. Pemilihan model aktif (aturan 1-SE)
// ---------------------------------------------------------------------------
function Pemilihan({ e }: { e: Evaluasi }) {
  const p = e.ringkasan_run.pemilihan_model;
  const label = (nama: string) => e.perbandingan_run.find((m) => m.nama_model === nama)?.label ?? nama;
  return (
    <Kartu
      id="pemilihan"
      judul="Bagaimana model aktif dipilih: aturan 1-SE"
      keterangan="Keputusan diambil hanya dari Cross Validation di data latih, bukan dari data uji, agar skor uji tetap jujur."
    >
      <ol className="list-decimal space-y-1 pl-5 text-sm text-tinta-2">
        <li>
          RMSE CV terkecil: <Angka>{label(p.model_rmse_terkecil)}</Angka> = <Angka>{formatEuro(p.rmse_terkecil, 2)}</Angka>.
        </li>
        <li>
          Standard error = std antar-fold ÷ √5 = <Angka>{formatEuro(p.standard_error, 2)}</Angka>, sehingga batas 1-SE ={" "}
          <Angka>{formatEuro(p.batas_1se, 2)}</Angka>.
        </li>
        <li>Model dengan RMSE CV di bawah batas dianggap setara secara statistik. Dari yang setara, dipilih yang paling sederhana.</li>
        <li>
          Terpilih: <Angka>{label(p.model_terpilih)}</Angka>.
        </li>
      </ol>
      <div className="mt-4 overflow-x-auto">
        <table className="tabel">
          <thead>
            <tr>
              <th>Model</th>
              <th className="kanan">RMSE CV (€)</th>
              <th className="kanan">Kompleksitas</th>
              <th>Dalam batas 1-SE?</th>
            </tr>
          </thead>
          <tbody>
            {p.kandidat.map((k) => (
              <tr key={k.model} className={k.model === p.model_terpilih ? "sorot" : ""}>
                <td>{label(k.model)}</td>
                <td className="kanan">{formatAngka(k.rmse_cv, 2)}</td>
                <td className="kanan">{k.kompleksitas}</td>
                <td>{k.dalam_batas_1se ? "✓ ya" : "✗ tidak"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-redup">Kompleksitas: 1 = regresi linear, 2 = Ridge (ada hyperparameter α), 3 = polinomial (lebih banyak bobot).</p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// D. Cross validation per fold
// ---------------------------------------------------------------------------
function CrossValidation({ e }: { e: Evaluasi }) {
  const cv = e.evaluasi.cv;
  const r = cv.ringkasan;
  const celah = r.rmse.rata_rata - r.rmse_latih.rata_rata;
  return (
    <Kartu
      id="cv"
      judul={`${cv.k}-Fold Cross Validation`}
      keterangan="Data latih dibagi 5 bagian; bergantian 4 bagian untuk melatih (preprocessing ikut di-fit ulang) dan 1 bagian untuk validasi."
    >
      <div className="overflow-x-auto">
        <table className="tabel">
          <thead>
            <tr>
              <th>Fold</th>
              <th className="kanan">RMSE latih (€)</th>
              <th className="kanan">RMSE validasi (€)</th>
              <th className="kanan">MAE (€)</th>
              <th className="kanan">MAPE</th>
              <th className="kanan">R²</th>
              <th className="kanan">R² log</th>
            </tr>
          </thead>
          <tbody>
            {cv.per_fold.map((f) => (
              <tr key={f.fold}>
                <td>{f.fold}</td>
                <td className="kanan">{formatAngka(f.rmse_latih, 1)}</td>
                <td className="kanan">{formatAngka(f.rmse, 1)}</td>
                <td className="kanan">{formatAngka(f.mae, 1)}</td>
                <td className="kanan">{formatPersen(f.mape, 2)}</td>
                <td className="kanan">{formatAngka(f.r2, 3)}</td>
                <td className="kanan">{formatAngka(f.r2_log, 3)}</td>
              </tr>
            ))}
            <tr className="font-semibold">
              <td>Rata-rata ± std</td>
              <td className="kanan">{formatAngka(r.rmse_latih.rata_rata, 1)}</td>
              <td className="kanan">
                {formatAngka(r.rmse.rata_rata, 1)} ± {formatAngka(r.rmse.std, 1)}
              </td>
              <td className="kanan">
                {formatAngka(r.mae.rata_rata, 1)} ± {formatAngka(r.mae.std, 1)}
              </td>
              <td className="kanan">
                {formatPersen(r.mape.rata_rata, 2)} ± {formatAngka(r.mape.std, 2)}
              </td>
              <td className="kanan">
                {formatAngka(r.r2.rata_rata, 3)} ± {formatAngka(r.r2.std, 3)}
              </td>
              <td className="kanan">
                {formatAngka(r.r2_log.rata_rata, 3)} ± {formatAngka(r.r2_log.std, 3)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-sm text-tinta-2">
        Selisih RMSE validasi dan latih rata-rata <Angka>{formatEuro(celah)}</Angka> (
        {formatPersen((celah / r.rmse_latih.rata_rata) * 100, 0)} dari RMSE latih).{" "}
        {celah / r.rmse_latih.rata_rata < 0.2
          ? "Selisih kecil: tidak ada tanda overfitting yang berarti."
          : "Selisih cukup besar: model mulai menghafal data latih (overfitting)."}{" "}
        Standar deviasi antar-fold ({formatEuro(r.rmse.std)}) menunjukkan seberapa besar skor bisa berubah hanya karena
        pembagian data yang berbeda.
      </p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// E. Prediksi vs aktual & residual
// ---------------------------------------------------------------------------
function PrediksiAktual({ e }: { e: Evaluasi }) {
  const titik = e.evaluasi.residual.titik;
  return (
    <Kartu
      id="prediksi-aktual"
      judul="Prediksi vs aktual, dan residual (data uji)"
      keterangan="Kiri: titik di garis putus-putus berarti tebakan tepat; di bawah garis = model menebak terlalu murah. Kanan: residual = ln(aktual) − ln(prediksi); pola acak di sekitar 0 tanpa corong berarti asumsi linearitas & variansi konstan terpenuhi."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <GrafikSebar
          titik={titik.map((t) => ({
            x: t.aktual_euro,
            y: t.prediksi_euro,
            judul: `${t.company} ${t.product}`,
            info: [["Tipe", t.typename], ["Selisih", formatEuro(t.residual_euro)]],
          }))}
          labelX="Harga aktual (€)"
          labelY="Harga prediksi (€)"
          garis="identitas"
        />
        <GrafikSebar
          titik={titik.map((t) => ({
            x: t.prediksi_euro,
            y: t.residual_log,
            judul: `${t.company} ${t.product}`,
            info: [["Aktual", formatEuro(t.aktual_euro)]],
          }))}
          labelX="Harga prediksi (€, skala log)"
          labelY="Residual ln(harga)"
          formatY={(v) => formatAngka(v, 1)}
          garis="nol"
          logX
        />
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// F. Normalitas residual
// ---------------------------------------------------------------------------
function Normalitas({ e }: { e: Evaluasi }) {
  const { histogram, statistik: s } = e.evaluasi.residual;
  const normal = s.shapiro_p >= 0.05;
  return (
    <Kartu
      id="normalitas"
      judul="Apakah residual berdistribusi normal?"
      keterangan="Regresi linear mengasumsikan error berdistribusi normal. Asumsi ini penting untuk rentang perkiraan harga (±1,96 × RMSE)."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-1 text-sm font-medium">Histogram residual ln(harga) + kurva normal</h3>
          <GrafikHistogram data={histogram} labelX="Residual ln(harga)" garisNormal />
        </div>
        <div>
          <h3 className="mb-1 text-sm font-medium">Q-Q plot</h3>
          <GrafikSebar
            titik={e.evaluasi.qq_plot.titik.map((t) => ({ x: t.teoretis, y: t.sampel }))}
            labelX="Kuantil teoretis normal"
            labelY="Kuantil residual terstandar"
            formatX={(v) => formatAngka(v, 1)}
            formatY={(v) => formatAngka(v, 1)}
            garis="diagonal"
            tinggi={260}
          />
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="tabel max-w-3xl">
          <tbody>
            <tr>
              <td>Uji Shapiro-Wilk (H₀: residual normal)</td>
              <td className="kanan">
                W = {formatAngka(s.shapiro_w, 4)}, p = {formatIlmiah(s.shapiro_p, 2)}
              </td>
              <td>{normal ? "p ≥ 0,05 → tidak ada bukti residual tidak normal" : "p < 0,05 → residual tidak normal sempurna"}</td>
            </tr>
            <tr>
              <td>Skewness (0 = simetris)</td>
              <td className="kanan">{formatAngka(s.skewness, 3)}</td>
              <td>{Math.abs(s.skewness) < 0.5 ? "hampir simetris" : s.skewness > 0 ? "condong kanan" : "condong kiri"}</td>
            </tr>
            <tr>
              <td>Excess kurtosis (0 = normal)</td>
              <td className="kanan">{formatAngka(s.kurtosis_excess, 3)}</td>
              <td>{s.kurtosis_excess > 0.5 ? "ekor lebih tebal: ada error ekstrem lebih sering dari normal" : "ekor mirip normal"}</td>
            </tr>
            <tr>
              <td>Korelasi |residual| dengan prediksi</td>
              <td className="kanan">{formatAngka(s.korelasi_abs_residual_vs_prediksi, 3)}</td>
              <td>
                {Math.abs(s.korelasi_abs_residual_vs_prediksi) < 0.2
                  ? "lemah: besar error tidak bergantung pada harga (homoskedastis)"
                  : "error membesar seiring harga (heteroskedastis)"}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// G. Validasi rentang perkiraan
// ---------------------------------------------------------------------------
function Interval({ e }: { e: Evaluasi }) {
  const i = e.evaluasi.metrik.interval;
  if (!i) return null;
  return (
    <Kartu
      id="interval"
      judul="Apakah rentang perkiraan harga bisa dipercaya?"
      keterangan="Halaman Prediksi menampilkan rentang exp(ln ŷ ± 1,96 × RMSE_log CV). Di sini klaim “~95%” diuji: berapa persen harga aktual data uji yang benar-benar jatuh di dalam rentangnya?"
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Target cakupan" nilai={formatPersen(i.target_cakupan_persen)} catatan={`z = ${formatAngka(i.z, 2)}`} />
        <StatTile label="Cakupan di data uji" nilai={formatPersen(i.cakupan_uji_persen)} catatan={`${i.jumlah_di_luar} dari ${i.jumlah_uji} di luar rentang`} />
        <StatTile label="RMSE log (CV)" nilai={formatAngka(i.rmse_log_cv, 3)} />
        <StatTile
          label="Lebar rentang"
          nilai={
            <span className="text-lg">
              {formatPersen(i.batas_bawah_relatif_persen, 0)} … {formatPersen(i.batas_atas_relatif_persen, 0, true)}
            </span>
          }
          catatan="relatif terhadap harga prediksi"
        />
      </div>
      <p className="mt-3 text-sm text-tinta-2">
        {Math.abs(i.cakupan_uji_persen - i.target_cakupan_persen) <= 3
          ? "Cakupan sesuai target, jadi rentang ini terkalibrasi dengan baik."
          : "Cakupan menyimpang dari target, jadi rentang perlu ditafsirkan hati-hati."}{" "}
        Rentangnya lebar dan tidak simetris dalam euro karena model bekerja pada skala log. Ini gambaran jujur ketidakpastian
        model yang hanya melihat spesifikasi.
      </p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// H. Learning curve
// ---------------------------------------------------------------------------
function KurvaBelajar({ e }: { e: Evaluasi }) {
  const lc = e.evaluasi.learning_curve;
  if (!lc) return <Kartu id="learning-curve" judul="Learning curve"><p className="text-sm text-tinta-2">Baseline tidak memiliki learning curve.</p></Kartu>;
  const akhir = lc[lc.length - 1];
  const sebelum = lc[lc.length - 3];
  return (
    <Kartu
      id="learning-curve"
      judul="Learning curve"
      keterangan="Error latih & validasi (rata-rata 5 fold) saat jumlah data latih ditambah sedikit demi sedikit."
    >
      <GrafikGaris
        data={lc.map((l) => ({ n: l.jumlah_data_latih, latih: l.rmse_latih, validasi: l.rmse_validasi }))}
        kunciX="n"
        seri={[
          { key: "latih", label: "RMSE latih", warna: SERI[0], titik: true },
          { key: "validasi", label: "RMSE validasi", warna: SERI[1], titik: true },
        ]}
        labelX="Jumlah data latih"
        labelY="RMSE (euro)"
        formatY={(v) => formatAngka(v, 0)}
      />
      <p className="mt-3 text-sm text-tinta-2">
        Dengan {formatAngka(akhir.jumlah_data_latih, 0)} data, RMSE latih <Angka>{formatEuro(akhir.rmse_latih)}</Angka> dan
        validasi <Angka>{formatEuro(akhir.rmse_validasi)}</Angka>. Dari {formatAngka(sebelum.jumlah_data_latih, 0)} ke{" "}
        {formatAngka(akhir.jumlah_data_latih, 0)} data, RMSE validasi berubah{" "}
        <Angka>{formatEuro(akhir.rmse_validasi - sebelum.rmse_validasi)}</Angka>.{" "}
        {akhir.rmse_validasi - sebelum.rmse_validasi < -5
          ? "Kurva validasi masih turun, jadi menambah data kemungkinan masih membantu."
          : "Kurva validasi sudah mendatar, jadi menambah data sejenis tidak banyak membantu. Perbaikan berikutnya perlu fitur yang lebih informatif."}
      </p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// I. Gradient descent
// ---------------------------------------------------------------------------
function GradientDescent({ gd }: { gd?: Evaluasi }) {
  const h = gd?.evaluasi.loss_history;
  if (!h) return <Kartu id="gradient-descent" judul="Kurva loss Gradient Descent"><StatusApi memuat teks="Memuat data Gradient Descent…" /></Kartu>;
  const semuaIterasi = Array.from(new Set(h.perbandingan_learning_rate.flatMap((k) => k.iterasi))).sort((a, b) => a - b);
  const data = semuaIterasi.map((it) => {
    const baris: Record<string, number | null> = { iterasi: it };
    h.perbandingan_learning_rate.forEach((k, j) => {
      const idx = k.iterasi.indexOf(it);
      baris[`lr${j}`] = idx >= 0 ? k.loss[idx] : null;
    });
    return baris;
  });
  return (
    <Kartu
      id="gradient-descent"
      judul="Kurva loss Gradient Descent"
      keterangan="MSE pada data latih di setiap iterasi untuk beberapa learning rate (sumbu Y logaritmik). Garis putus-putus = MSE solusi Normal Equation (titik terendah yang mungkin)."
    >
      <div className="mb-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatTile label="Learning rate dipakai" nilai={formatAngka(h.learning_rate, 2)} catatan={`batas stabil 2/λmaks = ${formatAngka(h.learning_rate_maks_stabil, 3)}`} />
        <StatTile label="Iterasi sampai konvergen" nilai={formatAngka(h.n_iterasi_dijalankan, 0)} catatan={`maksimum ${formatAngka(h.n_iterasi_maks, 0)}`} />
        <StatTile label="MSE akhir GD" nilai={formatAngka(h.loss_akhir, 6)} catatan={`awal ${formatAngka(h.loss_awal, 2)}`} />
        <StatTile label="MSE Normal Equation" nilai={formatAngka(h.loss_normal_equation, 6)} catatan={`selisih ${formatIlmiah(h.loss_akhir - h.loss_normal_equation)}`} />
      </div>
      <GrafikGaris
        data={data}
        kunciX="iterasi"
        seri={h.perbandingan_learning_rate.map((k, j) => ({ key: `lr${j}`, label: `lr = ${formatAngka(k.learning_rate, 2)}`, warna: SERI[j] }))}
        labelX="Iterasi"
        labelY="MSE (skala log)"
        formatY={(v) => formatAngka(v, v >= 1 ? 0 : 2)}
        logY
        garisAcuan={[{ y: h.loss_normal_equation, label: "Normal Equation" }]}
        tinggi={340}
      />
      <table className="tabel mt-4 max-w-2xl">
        <thead>
          <tr>
            <th>Learning rate</th>
            <th>Status</th>
            <th className="kanan">MSE di iterasi terakhir yang digambar</th>
          </tr>
        </thead>
        <tbody>
          {h.perbandingan_learning_rate.map((k) => (
            <tr key={k.learning_rate}>
              <td>{formatAngka(k.learning_rate, 2)}</td>
              <td>{k.status === "konvergen" ? "✓ konvergen" : "✗ divergen (loss meledak)"}</td>
              <td className="kanan">{k.loss_akhir === null ? "—" : formatAngka(k.loss_akhir, 5)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-3 text-sm text-tinta-2">
        Learning rate terlalu kecil membuat penurunan lambat; di atas batas {formatAngka(h.learning_rate_maks_stabil, 3)} (= 2 dibagi
        nilai eigen terbesar matriks Hessian {formatAngka(h.eigen_hessian_maks, 2)}) loss justru membesar terus. Kurva yang divergen
        dipotong setelah loss melewati 10× loss awal.
      </p>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// J. Verifikasi manual vs sklearn
// ---------------------------------------------------------------------------
function Verifikasi({ e }: { e: Evaluasi }) {
  return (
    <Kartu
      id="verifikasi"
      judul="Verifikasi: perhitungan manual vs scikit-learn"
      keterangan="Membuktikan bahwa rumus yang ditulis sendiri dengan NumPy menghasilkan angka yang sama dengan library standar."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Metrik model terpilih (data uji)</h3>
          <table className="tabel">
            <thead>
              <tr>
                <th>Metrik</th>
                <th className="kanan">Manual</th>
                <th className="kanan">sklearn.metrics</th>
                <th className="kanan">Selisih</th>
                <th>Sama?</th>
              </tr>
            </thead>
            <tbody>
              {e.evaluasi.verifikasi.map((v) => (
                <tr key={v.metrik}>
                  <td className="uppercase">{v.metrik}</td>
                  <td className="kanan">{formatAngka(v.manual, 6)}</td>
                  <td className="kanan">{formatAngka(v.sklearn, 6)}</td>
                  <td className="kanan">{formatIlmiah(v.selisih_absolut)}</td>
                  <td>{v.sama ? "✓" : "✗"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Koefisien (bobot) antar-implementasi</h3>
          <table className="tabel">
            <thead>
              <tr>
                <th>Perbandingan</th>
                <th className="kanan">Selisih maks bobot</th>
                <th className="kanan">Selisih maks prediksi</th>
              </tr>
            </thead>
            <tbody>
              {e.ringkasan_run.verifikasi_koefisien.map((v) => (
                <tr key={v.perbandingan}>
                  <td>
                    {v.perbandingan}
                    <div className="text-xs text-redup">{v.jumlah_koefisien} bobot + intersep</div>
                  </td>
                  <td className="kanan">{formatIlmiah(v.selisih_maks_koefisien)}</td>
                  <td className="kanan">{formatIlmiah(v.selisih_maks_prediksi_log)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-tinta-2">
            Selisih sekitar 10⁻¹³ hanyalah pembulatan floating point, artinya identik. Gradient Descent berselisih sekitar 10⁻⁵
            karena berhenti setelah jumlah iterasi tertentu (pendekatan iteratif), bukan solusi eksak.
          </p>
        </div>
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// K. Error per merek & tipe
// ---------------------------------------------------------------------------
function ErrorGrup({ e }: { e: Evaluasi }) {
  const merek = e.evaluasi.error_per_merek;
  const tipe = e.evaluasi.error_per_tipe;
  return (
    <Kartu
      id="error-grup"
      judul="Error per merek dan per tipe laptop (data uji)"
      keterangan="MAPE per kelompok. Kelompok dengan sedikit data (n kecil) hasilnya kurang stabil. Bias negatif = model cenderung menebak terlalu murah."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">MAPE per merek</h3>
          <GrafikBatangH
            lebarLabel={90}
            formatNilai={(v) => formatPersen(v, 0)}
            data={[...merek].sort((a, b) => b.mape - a.mape).map((g) => ({
              label: g.grup,
              nilai: g.mape,
              info: [
                ["Jumlah data uji", g.n],
                ["MAE", formatEuro(g.mae)],
                ["Bias rata-rata", formatEuro(g.bias_rata_rata)],
              ],
            }))}
          />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Per tipe laptop</h3>
          <table className="tabel">
            <thead>
              <tr>
                <th>Tipe</th>
                <th className="kanan">n</th>
                <th className="kanan">Rata-rata aktual</th>
                <th className="kanan">MAE</th>
                <th className="kanan">MAPE</th>
                <th className="kanan">Bias</th>
              </tr>
            </thead>
            <tbody>
              {tipe.map((g) => (
                <tr key={g.grup}>
                  <td>{g.grup}</td>
                  <td className="kanan">{g.n}</td>
                  <td className="kanan">{formatEuro(g.rata_rata_aktual)}</td>
                  <td className="kanan">{formatEuro(g.mae)}</td>
                  <td className="kanan">{formatPersen(g.mape)}</td>
                  <td className="kanan">{formatEuro(g.bias_rata_rata)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// L. 10 error terbesar
// ---------------------------------------------------------------------------
function TopError({ e }: { e: Evaluasi }) {
  const t = e.evaluasi.top_error;
  const murah = t.filter((x) => x.arah === "terlalu murah").length;
  return (
    <Kartu
      id="top-error"
      judul="10 prediksi dengan error terbesar (data uji)"
      keterangan={`${murah} dari ${t.length} ditebak terlalu murah. Laptop seperti ini biasanya membawa nilai yang tidak terekam di kolom spesifikasi (seri premium, GPU kelas atas, merek langka).`}
    >
      <div className="overflow-x-auto">
        <table className="tabel">
          <thead>
            <tr>
              <th>#</th>
              <th>Laptop</th>
              <th>Spesifikasi</th>
              <th className="kanan">Aktual</th>
              <th className="kanan">Prediksi</th>
              <th className="kanan">Error</th>
              <th>Arah</th>
            </tr>
          </thead>
          <tbody>
            {t.map((x) => (
              <tr key={x.peringkat}>
                <td>{x.peringkat}</td>
                <td>
                  {x.company} {x.product}
                  <div className="text-xs text-redup">{x.typename}</div>
                </td>
                <td className="text-xs text-tinta-2">
                  {x.cpu_type} · RAM {x.ram} GB · {x.memory} · {x.gpu_type}
                </td>
                <td className="kanan">{formatEuro(x.aktual_euro)}</td>
                <td className="kanan">{formatEuro(x.prediksi_euro)}</td>
                <td className="kanan">
                  {formatEuro(x.error_euro)}
                  <div className="text-xs text-redup">{formatPersen(x.error_persen, 0)}</div>
                </td>
                <td>{x.arah}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// M. Koefisien
// ---------------------------------------------------------------------------
function KoefisienModel({ e }: { e: Evaluasi }) {
  const k = e.evaluasi.koefisien;
  if (!k) return <Kartu id="koefisien" judul="Interpretasi koefisien"><p className="text-sm text-tinta-2">Baseline tidak memiliki koefisien.</p></Kartu>;
  const naik = k.filter((x) => x.koefisien > 0).slice(0, 5);
  const turun = [...k].reverse().filter((x) => x.koefisien < 0).slice(0, 5);
  const teratas = [...k].sort((a, b) => Math.abs(b.koefisien) - Math.abs(a.koefisien)).slice(0, 20);
  return (
    <Kartu
      id="koefisien"
      judul="Interpretasi koefisien: fitur apa yang menaikkan & menurunkan harga?"
      keterangan="Karena target = ln(harga), koefisien w berarti harga dikali e^w, yaitu berubah (e^w − 1) × 100%, dengan fitur lain tetap."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">Paling menaikkan harga</h3>
          <ul className="space-y-1 text-sm">
            {naik.map((x) => (
              <li key={x.fitur} className="flex justify-between gap-3 border-b border-garis py-1">
                <span>
                  {x.fitur}
                  <span className="block text-xs text-redup">{x.keterangan}</span>
                </span>
                <span className="angka font-medium text-biru-tua">{formatPersen(x.efek_persen, 1, true)}</span>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold">Paling menurunkan harga</h3>
          <ul className="space-y-1 text-sm">
            {turun.map((x) => (
              <li key={x.fitur} className="flex justify-between gap-3 border-b border-garis py-1">
                <span>
                  {x.fitur}
                  <span className="block text-xs text-redup">{x.keterangan}</span>
                </span>
                <span className="angka font-medium text-merah">{formatPersen(x.efek_persen, 1, true)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
      <div className="mt-6">
        <h3 className="mb-2 text-sm font-semibold">20 koefisien dengan pengaruh terbesar (efek terhadap harga, %)</h3>
        <GrafikBatangH
          divergen
          lebarLabel={200}
          formatNilai={(v) => formatPersen(v, 0, true)}
          data={teratas.map((x) => ({
            label: x.fitur,
            nilai: x.efek_persen,
            info: [["Koefisien w", formatAngka(x.koefisien, 4)], ["Arti", x.keterangan]],
          }))}
        />
      </div>
      <details className="mt-4">
        <summary className="cursor-pointer text-sm font-medium text-biru-tua">Tabel semua {k.length} koefisien</summary>
        <div className="mt-2 overflow-x-auto">
          <table className="tabel">
            <thead>
              <tr>
                <th>Fitur</th>
                <th>Jenis</th>
                <th className="kanan">Koefisien w</th>
                <th className="kanan">Efek (e^w − 1)</th>
                <th>Keterangan</th>
              </tr>
            </thead>
            <tbody>
              {k.map((x) => (
                <tr key={x.fitur}>
                  <td>{x.fitur}</td>
                  <td className="text-tinta-2">{x.jenis}</td>
                  <td className="kanan">{formatAngka(x.koefisien, 4)}</td>
                  <td className="kanan">{formatPersen(x.efek_persen, 1, true)}</td>
                  <td className="text-tinta-2">{x.keterangan}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// N. Multikolinearitas & Ridge
// ---------------------------------------------------------------------------
function Multikolinearitas({ e }: { e: Evaluasi }) {
  const m = e.ringkasan_run.multikolinearitas;
  const a = e.ringkasan_run.pemilihan_alpha;
  const vifPakai = new Map(m.vif_fitur_dipakai.map((v) => [v.fitur, v.vif]));
  const fiturLayar = ["Inches", "Lebar_px", "Tinggi_px", "PPI"];
  return (
    <Kartu
      id="multikolinearitas"
      judul="Multikolinearitas & Ridge Regression"
      keterangan="Multikolinearitas = beberapa fitur saling menjelaskan, sehingga bobotnya tidak stabil dan sulit ditafsirkan. Diukur dengan VIF (> 10 = tinggi) dan condition number matriks XᵀX."
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold">VIF fitur numerik (data latih)</h3>
          <table className="tabel">
            <thead>
              <tr>
                <th>Fitur</th>
                <th className="kanan">VIF semua fitur</th>
                <th className="kanan">VIF fitur dipakai</th>
              </tr>
            </thead>
            <tbody>
              {m.vif_semua_fitur_numerik.map((v) => (
                <tr key={v.fitur}>
                  <td>{v.fitur}</td>
                  <td className={`kanan ${v.vif > 10 ? "font-semibold text-merah" : ""}`}>{formatAngka(v.vif, 1)}</td>
                  <td className="kanan">{vifPakai.has(v.fitur) ? formatAngka(vifPakai.get(v.fitur)!, 1) : "dibuang"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="space-y-4">
          <div>
            <h3 className="mb-2 text-sm font-semibold">Condition number XᵀX per skenario</h3>
            <table className="tabel">
              <thead>
                <tr>
                  <th>Skenario</th>
                  <th className="kanan">Rank / kolom</th>
                  <th className="kanan">Condition number</th>
                </tr>
              </thead>
              <tbody>
                {m.skenario_condition_number.map((s) => (
                  <tr key={s.skenario}>
                    <td>{s.skenario}</td>
                    <td className="kanan">
                      {s.rank} / {s.jumlah_kolom}
                    </td>
                    <td className="kanan">{s.condition_number === null ? "∞ (singular)" : formatIlmiah(s.condition_number)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {m.kolom_identik_dibuang.length > 0 && (
            <KotakInfo judul="Kolinearitas sempurna yang terdeteksi otomatis">
              {m.kolom_identik_dibuang.map((k) => (
                <span key={k.dibuang}>
                  Kolom <b>{k.dibuang}</b> identik dengan <b>{k.identik_dengan}</b> di data latih, sehingga dibuang. Efek keduanya
                  tergabung di {k.identik_dengan}.{" "}
                </span>
              ))}
            </KotakInfo>
          )}
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="text-sm font-semibold">Jalur koefisien Ridge (lebar, tinggi, PPI, inci dipakai bersama)</h3>
          <p className="mb-2 text-xs text-tinta-2">Makin besar α, bobot fitur yang saling berkorelasi menyusut ke nol dan menjadi stabil. α = 0 sama dengan OLS.</p>
          <GrafikGaris
            data={m.jalur_koefisien_ridge_fitur_layar.map((j) => ({
              alpha: `α=${formatAngka(j.alpha, 0)}`,
              ...Object.fromEntries(fiturLayar.map((f) => [f, j[f]])),
            }))}
            kunciX="alpha"
            xKategori
            seri={fiturLayar.map((f, i) => ({ key: f, label: f, warna: SERI[i], titik: true }))}
            labelX="Kekuatan penalti"
            labelY="Bobot (fitur terstandar)"
            formatY={(v) => formatAngka(v, 2)}
            garisAcuan={[{ y: 0, label: "" }]}
          />
        </div>
        <div>
          <h3 className="text-sm font-semibold">Pemilihan α Ridge dengan 5-Fold CV</h3>
          <p className="mb-2 text-xs text-tinta-2">
            α terpilih = {formatAngka(a.alpha_terbaik, 3)}. Kurva datar untuk α kecil, artinya setelah fitur kolinear dibuang, Ridge
            hanya sedikit memperbaiki model.
          </p>
          <GrafikGaris
            data={a.kandidat.map((k) => ({ alpha: k.alpha, rmse: k.rmse_cv }))}
            kunciX="alpha"
            logX
            seri={[{ key: "rmse", label: "RMSE CV", warna: SERI[0], titik: true }]}
            labelX="α (skala log)"
            labelY="RMSE CV (euro)"
            formatX={(v) => formatAngka(v, v < 1 ? 3 : 0)}
            formatY={(v) => formatAngka(v, 0)}
            garisAcuan={[{ x: a.alpha_terbaik, label: "terpilih" }]}
          />
        </div>
      </div>
    </Kartu>
  );
}

// ---------------------------------------------------------------------------
// Halaman
// ---------------------------------------------------------------------------
export default function HalamanEvaluasi() {
  const daftar = useApi<DaftarModel>("/api/models");
  const [pilihan, setPilihan] = useState<number | null>(null);
  const idModel = pilihan ?? daftar.data?.model_aktif_id ?? daftar.data?.data[0]?.id ?? null;
  const evaluasi = useApi<Evaluasi>(idModel !== null ? `/api/evaluation/${idModel}` : null);
  const e = evaluasi.data;
  const idGD = e?.perbandingan_run.find((m) => m.nama_model === "ols_gradient_descent")?.id;
  const gd = useApi<Evaluasi>(idGD !== undefined && idGD !== idModel ? `/api/evaluation/${idGD}` : null);
  const dataGD = idGD === idModel ? e : gd.data;

  const versi = Array.from(new Set(daftar.data?.data.map((m) => m.nomor_versi) ?? []));
  const baseline = e?.perbandingan_run.find((m) => m.nama_model === "baseline_rata_rata");

  return (
    <div className="space-y-6">
      <JudulHalaman
        judul="Validasi & Evaluasi"
        keterangan="Bukti bahwa model benar-benar belajar dan perhitungan manualnya benar: pembanding baseline, cross validation, analisis residual, verifikasi terhadap scikit-learn, dan analisis error."
      />

      {!daftar.data ? (
        <StatusApi memuat={daftar.memuat} galat={daftar.galat} muatUlang={daftar.muatUlang} />
      ) : daftar.data.data.length === 0 ? (
        <KotakInfo jenis="peringatan" judul="Belum ada hasil training">
          Jalankan <code>python -m ml.train</code> di folder backend, lalu muat ulang halaman ini.
        </KotakInfo>
      ) : (
        <>
          <div className="sticky top-14 z-10 -mx-4 border-b border-garis bg-halaman/95 px-4 py-3 backdrop-blur">
            <div className="flex flex-wrap items-center gap-3">
              <label className="flex items-center gap-2 text-sm font-medium">
                Versi model
                <select
                  value={idModel ?? ""}
                  onChange={(ev) => setPilihan(Number(ev.target.value))}
                  className="max-w-[22rem] rounded-lg border border-sumbu bg-white px-2 py-1.5 text-sm"
                >
                  {versi.map((v) => {
                    const anggota = daftar.data!.data.filter((m) => m.nomor_versi === v);
                    return (
                      <optgroup key={v} label={`Versi ${v} · ${formatTanggal(anggota[0]?.dibuat_pada)}`}>
                        {anggota.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.label}
                            {m.is_active ? " (aktif)" : ""}
                          </option>
                        ))}
                      </optgroup>
                    );
                  })}
                </select>
              </label>
              <nav aria-label="Daftar isi" className="flex flex-1 gap-1 overflow-x-auto text-xs">
                {DAFTAR_ISI.map(([id, label]) => (
                  <a key={id} href={`#${id}`} className="whitespace-nowrap rounded-md px-2 py-1 text-tinta-2 hover:bg-permukaan hover:text-tinta">
                    {label}
                  </a>
                ))}
              </nav>
            </div>
          </div>

          {!e ? (
            <StatusApi memuat={evaluasi.memuat} galat={evaluasi.galat} muatUlang={evaluasi.muatUlang} />
          ) : (
            <div className="space-y-6">
              <KotakInfo judul={`${e.model.label} · versi ${e.ringkasan_run.nomor_versi}`}>
                {e.model.keterangan} Dilatih pada {formatTanggal(e.ringkasan_run.dibuat_pada)} dengan{" "}
                {formatAngka(e.ringkasan_run.dataset.jumlah_latih, 0)} data latih dan {formatAngka(e.ringkasan_run.dataset.jumlah_uji, 0)}{" "}
                data uji; {e.model.jumlah_fitur} fitur setelah one-hot encoding.
              </KotakInfo>
              <Ringkasan e={e} baseline={baseline} />
              <Perbandingan e={e} idTerpilih={e.model.id} />
              <Pemilihan e={e} />
              <CrossValidation e={e} />
              <PrediksiAktual e={e} />
              <Normalitas e={e} />
              <Interval e={e} />
              <KurvaBelajar e={e} />
              <GradientDescent gd={dataGD} />
              <Verifikasi e={e} />
              <ErrorGrup e={e} />
              <TopError e={e} />
              <KoefisienModel e={e} />
              <Multikolinearitas e={e} />
            </div>
          )}
        </>
      )}
    </div>
  );
}
