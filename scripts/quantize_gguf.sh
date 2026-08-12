#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-outputs/merged-router}"
OUT_DIR="${2:-outputs}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-external/llama.cpp}"

mkdir -p "$OUT_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "Missing $LLAMA_CPP_DIR. Clone llama.cpp first:" >&2
  echo "  git clone https://github.com/ggerganov/llama.cpp $LLAMA_CPP_DIR" >&2
  exit 1
fi

cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build"
cmake --build "$LLAMA_CPP_DIR/build" -j

python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_DIR" --outfile "$OUT_DIR/router-f16.gguf"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$OUT_DIR/router-f16.gguf" "$OUT_DIR/router-q4_k_m.gguf" Q4_K_M

echo "Wrote $OUT_DIR/router-q4_k_m.gguf"

