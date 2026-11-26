# Image backend: FastAPI + pipeline ML
FROM python:3.12-slim

# Python tidak menulis file .pyc & log langsung tampil (tidak ditahan buffer)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# 1) Pasang library dulu (lapisan ini di-cache selama requirements.txt tidak berubah)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) Salin kode, dataset, dan konfigurasi migrasi
COPY . .

# Jalankan sebagai user biasa (bukan root) demi keamanan
RUN useradd --create-home appuser && mkdir -p models outputs && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Saat container menyala: tunggu DB -> migrasi -> training jika perlu -> server API
CMD ["sh", "-c", "python siapkan.py && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
