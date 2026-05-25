#!/usr/bin/env bash
# Convenience helper that builds the pre-baked claude-code runner image.
#
# Usage:
#   bash evalscope/agent/external/runners/_assets/build_claude_code_image.sh
#
# Pass a custom tag as the first positional argument, e.g.:
#   bash .../build_claude_code_image.sh evalscope/claude-code:1.0
#
# The image is tagged ``evalscope/claude-code:dev`` by default so the
# bundled tests can find it without configuration.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKERFILE="${HERE}/claude_code.Dockerfile"
TAG="${1:-evalscope/claude-code:dev}"

if [[ ! -f "${DOCKERFILE}" ]]; then
    echo "Dockerfile not found at ${DOCKERFILE}" >&2
    exit 1
fi

# Empty context — the Dockerfile only needs base-image layers + apt/npm.
echo "Building ${TAG} from ${DOCKERFILE}..."
docker build -f "${DOCKERFILE}" -t "${TAG}" "${HERE}"
echo
echo "Built ${TAG}. Probe inside the image:"
docker run --rm "${TAG}" claude --version
