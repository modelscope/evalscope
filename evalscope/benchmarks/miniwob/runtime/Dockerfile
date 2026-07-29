# Build the pinned OpenEnv BrowserGym service with EvalScope's MiniWoB action-profile patch.
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-unifont \
    fonts-noto-color-emoji \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ARG OPENENV_COMMIT=65c506ef94bb1f7279cb4359673b3ef81031d01f
RUN git init /tmp/openenv && \
    git -C /tmp/openenv remote add origin https://github.com/huggingface/OpenEnv.git && \
    git -C /tmp/openenv fetch --depth 1 origin "${OPENENV_COMMIT}" && \
    git -C /tmp/openenv checkout --detach FETCH_HEAD

COPY openenv-miniwob-all.patch /tmp/openenv-miniwob-all.patch
RUN git -C /tmp/openenv apply --check /tmp/openenv-miniwob-all.patch && \
    git -C /tmp/openenv apply /tmp/openenv-miniwob-all.patch && \
    cp -a /tmp/openenv/envs/browsergym_env/. /app/env/ && \
    rm -rf /tmp/openenv /tmp/openenv-miniwob-all.patch

RUN chmod +x /app/env/server/start.sh

ARG EVALSCOPE_PIP_INDEX_URL=https://pypi.org/simple
RUN pip install --no-cache-dir --index-url "${EVALSCOPE_PIP_INDEX_URL}" \
    "fastmcp==3.0.0" \
    "gradio==6.20.0" \
    "openenv==0.4.1" \
    .

ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

ARG MINIWOB_COMMIT=7fd85d71a4b60325c6585396ec4f48377d049838
RUN mkdir -p /usr/local/share/miniwob-plusplus && \
    curl -L "https://github.com/Farama-Foundation/miniwob-plusplus/archive/${MINIWOB_COMMIT}.tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local/share/miniwob-plusplus && \
    chmod -R a+rX /usr/local/share/miniwob-plusplus

ENV PYTHONUNBUFFERED=1
ENV BROWSERGYM_BENCHMARK=miniwob
ENV BROWSERGYM_TASK_NAME="click-test"
ENV BROWSERGYM_HEADLESS=true
ENV BROWSERGYM_VIEWPORT_WIDTH=1280
ENV BROWSERGYM_VIEWPORT_HEIGHT=720
ENV BROWSERGYM_TIMEOUT=10000
ENV BROWSERGYM_PORT=8000
ENV MINIWOB_HTML_DIR=/usr/local/share/miniwob-plusplus/miniwob/html
ENV MINIWOB_HTTP_PORT=8888
ENV MINIWOB_URL=http://127.0.0.1:8888/miniwob/
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000
EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/app/env/server/start.sh"]
