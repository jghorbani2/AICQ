#!/usr/bin/env bash
set -euo pipefail

# Build a lightweight Lambda deployment bundle containing only the Bedrock
# classifier and minimal handler. The output directory defaults to
# dist/lambda-minimal and a matching ZIP is created for upload.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${1:-"$ROOT_DIR/dist/lambda-minimal"}"
ZIP_PATH="${OUT_DIR}.zip"

echo "Building minimal Lambda bundle in: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

# Copy handler
cp "$ROOT_DIR/lambda_handler.py" "$OUT_DIR/"

# Copy only the lightweight Bedrock client artefacts
rsync -a \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude "requirements.txt" \
  --exclude "template.yaml" \
  "$ROOT_DIR/aws_bedrock_agent" \
  "$OUT_DIR/"

# Remove optional heavy assets not needed for classification
rm -f "$OUT_DIR/aws_bedrock_agent"/{data_pipeline.py,pipeline_runner.py,app.py,agent_invoke.py,agentcore_setup.py,requirements.txt,template.yaml}

# Zip the bundle
rm -f "$ZIP_PATH"
(
  cd "$(dirname "$OUT_DIR")"
  zip -r "$(basename "$ZIP_PATH")" "$(basename "$OUT_DIR")" >/dev/null
)

echo "Bundle ready: $ZIP_PATH"
echo "Size:"
du -h "$ZIP_PATH"
