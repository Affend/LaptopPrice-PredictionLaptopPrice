# Image frontend: Next.js (build multi-tahap agar image akhir kecil)

# ---------- Tahap 1: pasang dependensi ----------
FROM node:24-alpine AS dependensi
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---------- Tahap 2: build ----------
FROM node:24-alpine AS build
WORKDIR /app
COPY --from=dependensi /app/node_modules ./node_modules
COPY . .
# Alamat API yang dipanggil BROWSER. Nilai NEXT_PUBLIC_* ditanam saat build,
# jadi harus diberikan di sini (lewat build arg di docker-compose.yml).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# ---------- Tahap 3: image siap jalan ----------
FROM node:24-alpine AS jalan
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
# Hanya hasil build "standalone" + file statis yang disalin
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
