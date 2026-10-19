# DS-1000 official evaluation environment (environment.yml of https://github.com/xlang-ai/DS-1000).
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==1.5.3 \
    scipy==1.12.0 \
    scikit-learn==1.4.0 \
    matplotlib==3.8.4 \
    seaborn==0.13.2 \
    statsmodels==0.14.1 \
    xgboost==2.0.3 \
    gensim==4.3.2 \
    pyyaml

# PyPI aarch64 torch wheels are CPU-only; x86_64 wheels bundle CUDA, so use the CPU index there.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.2.0; \
    fi

# tensorflow-cpu publishes no linux/aarch64 wheel; the Tensorflow subset only runs on linux/amd64.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --no-cache-dir tensorflow-cpu==2.16.1; \
    fi
