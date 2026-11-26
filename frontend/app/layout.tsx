import type { Metadata } from "next";
import Navigasi from "@/components/Navigasi";
import "./globals.css";

export const metadata: Metadata = {
  title: "LaptopPrice — Prediksi Harga Laptop",
  description:
    "Sistem prediksi harga laptop berdasarkan spesifikasi menggunakan Regresi Linear Berganda (Tugas Besar Kecerdasan Buatan).",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="id" className="h-full antialiased">
      <body className="flex min-h-full flex-col">
        <Navigasi />
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:py-8">{children}</main>
        <footer className="border-t border-garis py-4 text-center text-xs text-redup">
          LaptopPrice · Tugas Besar Kecerdasan Buatan · Regresi Linear Berganda
        </footer>
      </body>
    </html>
  );
}
