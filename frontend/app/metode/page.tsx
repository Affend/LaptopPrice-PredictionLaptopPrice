import "katex/dist/katex.min.css";
import type { ReactNode } from "react";
import Rumus from "@/components/Rumus";
import { JudulHalaman, Kartu } from "@/components/ui";

/**
 * Halaman Penjelasan Metode — hanya rumus & penjelasan konsep.
 * Tidak berisi angka hasil (angka hasil ada di halaman Validasi & Evaluasi).
 */

function Bagian({ id, judul, children }: { id: string; judul: string; children: ReactNode }) {
  return (
    <Kartu id={id} judul={judul}>
      <div className="max-w-3xl space-y-3 leading-relaxed text-tinta-2 [&_b]:text-tinta">{children}</div>
    </Kartu>
  );
}

const ISI = [
  ["regresi", "Regresi linear berganda"],
  ["log", "Mengapa ln(harga)?"],
  ["mse", "Fungsi loss (MSE)"],
  ["normal-equation", "Normal Equation"],
  ["gradient-descent", "Gradient Descent"],
  ["ridge", "Ridge & polinomial"],
  ["preprocessing", "Standardisasi & one-hot"],
  ["metrik", "Metrik evaluasi"],
  ["validasi", "Skema validasi"],
  ["multikolinearitas", "Multikolinearitas"],
  ["rentang", "Rentang perkiraan & penyesuaian harga"],
] as const;

export default function HalamanMetode() {
  return (
    <div className="space-y-6">
      <JudulHalaman
        judul="Penjelasan Metode"
        keterangan="Rumus-rumus yang dipakai sistem ini, dijelaskan dengan bahasa sederhana. Semua rumus diimplementasikan di backend/ml/ (model.py dan evaluation.py)."
      />
      <nav aria-label="Daftar isi" className="flex flex-wrap gap-2 text-sm">
        {ISI.map(([id, label]) => (
          <a key={id} href={`#${id}`} className="rounded-lg border border-garis bg-permukaan px-3 py-1 text-tinta-2 hover:border-biru hover:text-tinta">
            {label}
          </a>
        ))}
      </nav>

      <Bagian id="regresi" judul="1. Regresi linear berganda">
        <p>
          Regresi linear berganda menebak sebuah nilai (di sini: harga laptop) sebagai <b>jumlah berbobot</b> dari beberapa fitur
          (RAM, ukuran SSD, merek, dan lain-lain), ditambah satu konstanta:
        </p>
        <Rumus blok tex={String.raw`\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_p x_p`} />
        <p>
          <Rumus tex="x_j" /> adalah nilai fitur ke-<i>j</i>, <Rumus tex="w_j" /> bobotnya (seberapa besar pengaruh fitur itu), dan{" "}
          <Rumus tex="w_0" /> intersep (harga dasar ketika semua fitur bernilai nol). Untuk <i>n</i> laptop sekaligus, rumus ini
          ditulis dalam bentuk matriks:
        </p>
        <Rumus blok tex={String.raw`\hat{\mathbf{y}} = X\mathbf{w}, \qquad X = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1p} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n1} & \cdots & x_{np}\end{bmatrix}`} />
        <p>Kolom angka 1 di kiri matriks <i>X</i> dikalikan dengan <Rumus tex="w_0" />, sehingga intersep ikut terhitung dalam perkalian matriks.</p>
      </Bagian>

      <Bagian id="log" judul="2. Mengapa memprediksi ln(harga)?">
        <p>Model tidak menebak harga langsung, melainkan logaritma natural harga:</p>
        <Rumus blok tex={String.raw`y = \ln(\text{harga}) \quad\Longrightarrow\quad \widehat{\text{harga}} = e^{\hat{y}}`} />
        <ul className="list-disc space-y-1 pl-5">
          <li>Harga laptop miring ke kanan (banyak yang murah, sedikit yang sangat mahal); setelah di-log distribusinya hampir simetris.</li>
          <li>Pengaruh spesifikasi terhadap harga bersifat <b>persentase</b>, bukan nominal tetap. Di skala log, efek perkalian menjadi penjumlahan, sehingga cocok untuk model linear.</li>
          <li>Hasil <Rumus tex="e^{\hat{y}}" /> selalu positif, jadi tidak mungkin memprediksi harga negatif.</li>
        </ul>
        <p>Akibatnya, setiap bobot dibaca sebagai persentase perubahan harga jika fitur naik satu satuan (fitur lain tetap):</p>
        <Rumus blok tex={String.raw`\text{perubahan harga} = \left(e^{w_j} - 1\right) \times 100\%`} />
        <p>
          RAM dan kapasitas penyimpanan juga dimasukkan dalam bentuk <Rumus tex={String.raw`\log_2`} />, sehingga bobotnya berarti
          “efek setiap kali kapasitas dua kali lipat”.
        </p>
      </Bagian>

      <Bagian id="mse" judul="3. Fungsi loss: Mean Squared Error">
        <p>
          Bobot terbaik adalah bobot yang membuat tebakan paling dekat dengan harga sebenarnya. “Dekat” diukur dengan rata-rata
          kuadrat selisih (<b>MSE</b>):
        </p>
        <Rumus blok tex={String.raw`J(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2 = \frac{1}{n}\lVert \mathbf{y} - X\mathbf{w}\rVert^2`} />
        <p>Selisih dikuadratkan agar error positif dan negatif tidak saling meniadakan, dan agar error besar dihukum lebih berat.</p>
      </Bagian>

      <Bagian id="normal-equation" judul="4. Normal Equation (solusi langsung)">
        <p>
          MSE berbentuk mangkuk (fungsi kuadrat), sehingga titik terendahnya ada di tempat turunannya sama dengan nol. Turunan
          terhadap <b>w</b>:
        </p>
        <Rumus blok tex={String.raw`\nabla_{\mathbf{w}} J = -\frac{2}{n} X^\top(\mathbf{y} - X\mathbf{w}) = 0 \;\Longrightarrow\; X^\top X\,\mathbf{w} = X^\top \mathbf{y}`} />
        <Rumus blok tex={String.raw`\boxed{\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{y}}`} />
        <p>
          Jika ada fitur yang saling bergantung sempurna, <Rumus tex={String.raw`X^\top X`} /> tidak punya invers (singular). Karena itu
          kode memakai <b>pseudo-inverse</b> <Rumus tex={String.raw`(X^\top X)^{+}`} /> (dihitung lewat SVD), yang tetap memberi solusi.
        </p>
      </Bagian>

      <Bagian id="gradient-descent" judul="5. Gradient Descent (solusi bertahap)">
        <p>
          Alih-alih menghitung invers, bobot dimulai dari nol lalu digeser sedikit demi sedikit ke arah yang menurunkan loss,
          seperti menuruni bukit dengan langkah kecil:
        </p>
        <Rumus blok tex={String.raw`\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \eta\, \nabla_{\mathbf{w}} J, \qquad \nabla_{\mathbf{w}} J = \frac{2}{n} X^\top\!\left(X\mathbf{w}^{(t)} - \mathbf{y}\right)`} />
        <p>
          <Rumus tex={String.raw`\eta`} /> adalah <b>learning rate</b> (besar langkah). Terlalu kecil berarti lambat; terlalu besar
          membuat loss meloncat-loncat lalu meledak. Untuk MSE, langkah dijamin stabil jika
        </p>
        <Rumus blok tex={String.raw`\eta < \frac{2}{\lambda_{\max}}, \qquad \lambda_{\max} = \text{nilai eigen terbesar dari } \tfrac{2}{n}X^\top X`} />
        <p>
          Fitur distandardisasi lebih dulu agar “bukit”-nya tidak terlalu lonjong, sehingga Gradient Descent cepat konvergen. Hasil
          akhirnya harus sama (sampai beberapa angka desimal) dengan Normal Equation, dan ini diverifikasi di halaman Validasi.
        </p>
      </Bagian>

      <Bagian id="ridge" judul="6. Ridge Regression dan regresi polinomial">
        <p>Ridge menambahkan penalti terhadap besarnya bobot agar bobot tidak “liar” saat fitur saling berkorelasi:</p>
        <Rumus blok tex={String.raw`J_{\text{ridge}}(\mathbf{w}) = \lVert \mathbf{y} - X\mathbf{w}\rVert^2 + \alpha \sum_{j=1}^{p} w_j^2 \;\Longrightarrow\; \mathbf{w} = (X^\top X + \alpha I')^{-1} X^\top\mathbf{y}`} />
        <p>
          <Rumus tex="I'" /> adalah matriks identitas dengan elemen intersep dibuat 0 (intersep tidak dipenalti). Makin besar{" "}
          <Rumus tex={String.raw`\alpha`} />, bobot makin menyusut ke nol. Nilai <Rumus tex={String.raw`\alpha`} /> dipilih lewat cross validation.
        </p>
        <p>
          <b>Regresi polinomial derajat 2</b> tetap regresi linear (linear terhadap bobot), tetapi fiturnya ditambah kuadrat dan
          perkalian antar-fitur numerik utama, sehingga bisa menangkap hubungan yang melengkung:
        </p>
        <Rumus blok tex={String.raw`\hat{y} = w_0 + \sum_j w_j x_j + \sum_{j \le k} w_{jk}\, x_j x_k`} />
      </Bagian>

      <Bagian id="preprocessing" judul="7. Standardisasi, one-hot encoding, dan data leakage">
        <p>Fitur numerik diubah ke skala yang sama (rata-rata 0, standar deviasi 1):</p>
        <Rumus blok tex={String.raw`z = \frac{x - \mu_{\text{latih}}}{\sigma_{\text{latih}}}`} />
        <p>
          <Rumus tex={String.raw`\mu`} /> dan <Rumus tex={String.raw`\sigma`} /> dihitung <b>hanya dari data latih</b>. Jika data uji
          ikut dipakai, model secara tidak langsung “mengintip” data uji (<b>data leakage</b>) dan skornya menjadi terlalu optimis.
        </p>
        <p>
          Fitur kategori (merek, tipe, CPU, GPU, OS) diubah menjadi kolom 0/1 (<b>one-hot encoding</b>). Satu kategori acuan per
          fitur sengaja tidak diberi kolom (<b>drop-first</b>). Tanpa itu, jumlah kolom one-hot selalu 1 untuk setiap baris, sama
          persis dengan kolom intersep, sehingga terjadi kolinearitas sempurna (<i>dummy variable trap</i>). Bobot kategori dibaca
          sebagai selisih terhadap kategori acuan.
        </p>
      </Bagian>

      <Bagian id="metrik" judul="8. Metrik evaluasi">
        <p>Semua metrik dihitung pada skala harga asli (euro), setelah prediksi dikembalikan dengan <Rumus tex="e^{\hat{y}}" />.</p>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="rounded-lg border border-garis bg-white p-3">
            <b>MAE</b>: rata-rata besar kesalahan, dalam euro.
            <Rumus blok tex={String.raw`\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|`} />
          </div>
          <div className="rounded-lg border border-garis bg-white p-3">
            <b>RMSE</b>: seperti MAE, tetapi error besar lebih dihukum. Selalu ≥ MAE.
            <Rumus blok tex={String.raw`\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}`} />
          </div>
          <div className="rounded-lg border border-garis bg-white p-3">
            <b>MAPE</b>: rata-rata kesalahan dalam persen terhadap harga asli.
            <Rumus blok tex={String.raw`\text{MAPE} = \frac{100\%}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|`} />
          </div>
          <div className="rounded-lg border border-garis bg-white p-3">
            <b>R²</b>: proporsi variasi harga yang dijelaskan model. 1 = sempurna, 0 = sama dengan menebak rata-rata.
            <Rumus blok tex={String.raw`R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}`} />
          </div>
        </div>
        <p>
          Keempat rumus ditulis manual dengan NumPy, lalu dibandingkan dengan <code>sklearn.metrics</code>; hasilnya harus sama
          persis. Semua model juga dibandingkan dengan <b>baseline</b> yang selalu menebak rata-rata harga.
        </p>
      </Bagian>

      <Bagian id="validasi" judul="9. Skema validasi">
        <ol className="list-decimal space-y-2 pl-5">
          <li>
            <b>Split 80/20</b> dengan <code>random_state</code> tetap: 20% data disimpan sebagai data uji dan tidak disentuh sampai
            akhir.
          </li>
          <li>
            <b>5-Fold Cross Validation</b> pada data latih: data latih dibagi 5; bergantian 4 bagian melatih dan 1 bagian menguji.
            Hasilnya rata-rata ± standar deviasi dari 5 skor.
          </li>
          <li>
            Semua keputusan (memilih <Rumus tex={String.raw`\alpha`} />, memilih model aktif) diambil dari CV. Model aktif memakai{" "}
            <b>aturan 1-SE</b>: model paling sederhana yang RMSE CV-nya masih dalam satu standard error dari yang terbaik.
            <Rumus blok tex={String.raw`\text{SE} = \frac{s_{\text{RMSE antar-fold}}}{\sqrt{k}}, \qquad \text{batas} = \text{RMSE}_{\min} + \text{SE}`} />
          </li>
          <li>Data uji dipakai sekali di akhir untuk melaporkan performa yang jujur.</li>
        </ol>
        <p>
          Diagnostik tambahan: grafik residual (pola acak = asumsi linear terpenuhi), Q-Q plot & uji Shapiro-Wilk (normalitas
          residual), learning curve (overfitting atau underfitting), serta analisis error per merek/tipe.
        </p>
      </Bagian>

      <Bagian id="multikolinearitas" judul="10. Multikolinearitas: VIF dan condition number">
        <p>
          Jika satu fitur bisa ditebak dari fitur lain (mis. lebar resolusi dari tinggi resolusi), bobotnya menjadi tidak stabil.
          Ukurannya <b>Variance Inflation Factor</b>:
        </p>
        <Rumus blok tex={String.raw`\text{VIF}_j = \frac{1}{1 - R_j^2}`} />
        <p>
          <Rumus tex="R_j^2" /> adalah R² ketika fitur <i>j</i> diregresikan terhadap fitur lain. VIF di atas 10 dianggap tinggi.
          Ukuran lain adalah <b>condition number</b> <Rumus tex={String.raw`\kappa = \lambda_{\max} / \lambda_{\min}`} /> dari{" "}
          <Rumus tex={String.raw`X^\top X`} />: makin besar, makin rapuh inversnya. Solusinya membuang fitur yang berlebihan atau
          memakai Ridge.
        </p>
      </Bagian>

      <Bagian id="rentang" judul="11. Rentang perkiraan dan penyesuaian harga">
        <p>
          Jika residual di skala log kira-kira normal dengan simpangan <Rumus tex={String.raw`\text{RMSE}_{\log}`} /> (dari CV),
          sekitar 95% harga sebenarnya jatuh di dalam
        </p>
        <Rumus blok tex={String.raw`\left[\, e^{\hat{y} - 1{,}96\,\text{RMSE}_{\log}},\; e^{\hat{y} + 1{,}96\,\text{RMSE}_{\log}} \,\right]`} />
        <p>Klaim 95% ini diuji pada data uji di halaman Validasi.</p>
        <p>
          Harga di dataset berada pada tingkat harga sekitar tahun 2018. Untuk ditampilkan dalam rupiah saat ini, hasil prediksi
          dikalikan faktor inflasi zona euro (indeks HICP Eurostat) dan kurs:
        </p>
        <Rumus blok tex={String.raw`\text{harga}_{\text{Rp}} = \widehat{\text{harga}}_{\text{€, 2018}} \times \frac{\text{HICP}_{\text{terkini}}}{\overline{\text{HICP}}_{2018}} \times \text{kurs}_{\text{€}\to\text{Rp}}`} />
        <p>
          Penyesuaian ini menyetarakan nilai uang, bukan harga pasar laptop baru: laptop dengan spesifikasi setara 2018 hari ini
          umumnya lebih murah karena teknologinya sudah tua. Faktor ini tidak mengubah MAPE dan R², karena semua harga dikali
          bilangan yang sama.
        </p>
      </Bagian>
    </div>
  );
}
