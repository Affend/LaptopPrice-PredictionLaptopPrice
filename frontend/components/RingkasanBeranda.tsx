"use client";

import { useApi } from "@/lib/api";
import { formatAngka, formatPersen } from "@/lib/format";
import type { DaftarModel, Health } from "@/lib/tipe";
import { StatTile, StatusApi } from "./ui";

/** Angka ringkas di beranda — semuanya diambil dari API, bukan ditulis manual. */
export default function RingkasanBeranda() {
  const kesehatan = useApi<Health>("/api/health");
  const model = useApi<DaftarModel>("/api/models");

  if (!kesehatan.data || !model.data) {
    return (
      <StatusApi
        memuat={kesehatan.memuat || model.memuat}
        galat={kesehatan.galat ?? model.galat}
        muatUlang={() => {
          kesehatan.muatUlang();
          model.muatUlang();
        }}
      />
    );
  }
  const aktif = model.data.data.find((m) => m.id === model.data?.model_aktif_id);
  const baseline = model.data.data.find(
    (m) => m.nama_model === "baseline_rata_rata" && m.nomor_versi === aktif?.nomor_versi,
  );
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      <StatTile label="Data laptop dipakai" nilai={formatAngka(kesehatan.data.jumlah_data, 0)} catatan="setelah pembersihan" />
      <StatTile
        label="Model aktif"
        nilai={<span className="text-base leading-snug">{aktif?.label ?? "Belum ada"}</span>}
        catatan={aktif ? `versi ${aktif.nomor_versi}` : kesehatan.data.pesan_model}
      />
      <StatTile
        label="MAPE data uji"
        nilai={aktif ? formatPersen(aktif.metrik_uji.mape) : "-"}
        catatan={baseline ? `baseline rata-rata: ${formatPersen(baseline.metrik_uji.mape)}` : undefined}
      />
      <StatTile
        label="R² data uji (skala log)"
        nilai={aktif ? formatAngka(aktif.metrik_uji.r2_log, 3) : "-"}
        catatan={aktif ? `R² skala euro: ${formatAngka(aktif.metrik_uji.r2, 3)}` : undefined}
      />
    </div>
  );
}
