import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // "standalone": next build menghasilkan folder .next/standalone berisi server.js
  // dan hanya file node_modules yang dibutuhkan -> image Docker jauh lebih kecil.
  output: "standalone",
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
