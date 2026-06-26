FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git \
    ffmpeg \
    fonts-dejavu-core \
    ghostscript \
    libreoffice-calc libreoffice-core libreoffice-impress libreoffice-writer \
    poppler-utils \
    tesseract-ocr \
    unzip zip \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
    matplotlib numpy openpyxl pandas pdfplumber pillow python-docx python-pptx reportlab scipy xlrd

WORKDIR /workspace
RUN mkdir -p deliverable_files

CMD ["/bin/bash"]
