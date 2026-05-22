# Pre-baked claude-code runner image.
#
# Used by benchmarks where the per-sample sandbox does NOT come with a
# fixed working tree (e.g. gsm8k, generic single-turn evals running
# through the external agent path). For SWE-bench / SWE-bench_Pro this
# image is irrelevant — those benchmarks pin a per-instance sweap-image
# and the runner does runtime install inside it.
#
# Build:
#   docker build -f evalscope/agent/external/runners/_assets/claude_code.Dockerfile \
#                -t evalscope/claude-code:dev .
#
# Or via the bundled helper:
#   bash evalscope/agent/external/runners/_assets/build_claude_code_image.sh
#
# Pin ``python:3.11-slim`` to match other evalscope tooling; image is
# multi-arch so it works on Apple Silicon / Linux amd64 alike. The npm
# package is intentionally NOT version-pinned because claude-code is
# updated frequently and the npm cache already pins to whatever was
# resolved at build time.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    NPM_CONFIG_FUND=false \
    NPM_CONFIG_AUDIT=false

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git \
 && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
 && apt-get install -y --no-install-recommends nodejs \
 && npm install -g @anthropic-ai/claude-code \
 && apt-get purge -y --auto-remove curl gnupg \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /root/.npm

# Smoke-test the install so a broken image fails at build time, not at
# the first sample of a 4-hour SWE-bench run.
RUN node --version && npm --version && claude --version

WORKDIR /workspace
