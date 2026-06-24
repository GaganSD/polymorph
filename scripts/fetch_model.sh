#!/usr/bin/env bash
# Download the optional LaMR ONNX model bundle used by POLYMORPH_LAMR_MODEL.
#
# Defaults target the public GitHub Release asset convention. Maintainers should
# upload a tarball containing at least model.onnx, and optionally model.onnx.data
# plus decode.json, then set POLYMORPH_MODEL_SHA256 below or pass it via env.
#
# Overrides:
#   POLYMORPH_MODEL_URL     direct URL to a .tar.gz/.tgz/.zip archive or model.onnx
#   POLYMORPH_MODEL_SHA256  expected SHA256 of the downloaded file
#   POLYMORPH_MODEL_OUT     output dir, default data/modal_out/mb_v0/onnx

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${POLYMORPH_MODEL_VERSION:-mb_v0}"
DEFAULT_URL="https://github.com/GaganSD/lulu-polymorph/releases/download/${VERSION}/polymorph-${VERSION}-onnx.tar.gz"
URL="${POLYMORPH_MODEL_URL:-$DEFAULT_URL}"
SHA256="${POLYMORPH_MODEL_SHA256:-}"
OUT_DIR="${POLYMORPH_MODEL_OUT:-$ROOT/data/modal_out/mb_v0/onnx}"

mkdir -p "$OUT_DIR"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

case "$URL" in
  *.zip) DOWNLOAD="$TMP_DIR/model.zip" ;;
  *.onnx) DOWNLOAD="$TMP_DIR/model.onnx" ;;
  *) DOWNLOAD="$TMP_DIR/model.tar.gz" ;;
esac

echo "[model] downloading:"
echo "  $URL"
echo "[model] output dir:"
echo "  $OUT_DIR"

curl -fL --retry 3 --retry-delay 2 -o "$DOWNLOAD" "$URL"

if [ -n "$SHA256" ]; then
  echo "$SHA256  $DOWNLOAD" | shasum -a 256 -c -
else
  echo "[model] warning: POLYMORPH_MODEL_SHA256 is not set; checksum verification skipped"
fi

case "$DOWNLOAD" in
  *.zip)
    unzip -o -q "$DOWNLOAD" -d "$OUT_DIR"
    ;;
  *.onnx)
    cp "$DOWNLOAD" "$OUT_DIR/model.onnx"
    ;;
  *)
    tar -xzf "$DOWNLOAD" -C "$OUT_DIR"
    ;;
esac

if [ ! -f "$OUT_DIR/model.onnx" ]; then
  FOUND="$(find "$OUT_DIR" -name model.onnx -type f 2>/dev/null | head -1 || true)"
  if [ -n "$FOUND" ]; then
    cp "$FOUND" "$OUT_DIR/model.onnx"
  fi
fi

if [ ! -f "$OUT_DIR/model.onnx" ]; then
  echo "[model] error: expected model.onnx under $OUT_DIR after extraction" >&2
  exit 1
fi

echo "[model] installed:"
ls -lh "$OUT_DIR"/model.onnx "$OUT_DIR"/model.onnx.data "$OUT_DIR"/decode.json 2>/dev/null || true
echo
echo "Set this in your MCP config:"
echo "  POLYMORPH_LAMR_MODEL=$OUT_DIR/model.onnx"
