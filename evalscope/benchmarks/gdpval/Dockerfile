# Slim base, good glibc; Python 3.11 fits most wheels on PyPI
# Force AMD64 platform for maximum package compatibility
FROM --platform=linux/amd64 python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# --- System runtime deps (carefully curated) ---
# - ffmpeg + dev libs: moviepy/ffmpeg-python/pydub/av (av package needs dev headers)
# - libsndfile1: soundfile/librosa
# - tesseract-ocr: pytesseract
# - poppler-utils: pdf2image
# - wkhtmltopdf: pdfkit/imgkit
# - ghostscript + default-jre: camelot, tabula
# - graphviz + libgraphviz-dev: graphviz CLI + build headers for pygraphviz
# - zbar: pyzbar
# - libcairo2, pango, gdk-pixbuf, libxml2, libxslt, libffi: cairosvg/weasyprint
# - libgl1, libglib2.0-0, libsm6, libxext6, libxrender1: OpenCV/gui-less rendering
# - fonts-dejavu-core: sane default fonts for matplotlib/wordcloud
# - LibreOffice (headless-friendly subsets)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    ffmpeg \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    libsndfile1 \
    tesseract-ocr \
    poppler-utils \
    wkhtmltopdf \
    ghostscript \
    default-jre \
    graphviz libgraphviz-dev \
    libzbar0 \
    libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf-2.0-0 libffi8 libxml2 libxslt1.1 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    fonts-dejavu-core \
    libreoffice-core libreoffice-writer libreoffice-calc libreoffice-impress \
    build-essential cmake pkg-config \
    libgeos-c1v5 libgeos-dev \
    libopenblas-dev liblapack-dev \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools for better wheel support
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Prefer CPU wheels for PyTorch (keeps image smaller and avoids CUDA in slim)
RUN python -m pip config set global.index-url https://pypi.org/simple \
    && python -m pip config set global.extra-index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install
WORKDIR /workspace
RUN mkdir deliverable_files
COPY docker-requirements.txt /tmp/docker-requirements.txt

# Install Python packages (build tools needed for some packages)
RUN python -m pip install --no-cache-dir -r /tmp/docker-requirements.txt

# Clean up build tools after installation (keep graphviz/ffmpeg runtime)
RUN apt-get purge -y build-essential cmake pkg-config libgraphviz-dev \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip /tmp/*

# Create a non-root user
RUN useradd -ms /bin/bash appuser

# Give appuser ownership of workspace
RUN chown -R appuser:appuser /workspace

USER appuser
ENV HOME=/home/appuser

# Jupyter defaults
EXPOSE 8888
CMD ["/bin/bash"]
