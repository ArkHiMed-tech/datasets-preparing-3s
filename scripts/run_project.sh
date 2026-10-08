#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_NAME="diabetes_best_model_pipeline"
DATA_PATH="${PROJECT_ROOT}/data/raw"
RANDOM_SEED=42
EPOCHS=100
OUTPUT_DIR="${PROJECT_ROOT}"

usage() {
    cat <<'EOF'
Usage: scripts/run_project.sh [options]

Options:
  --model-name NAME       Model name without extension (default: diabetes_best_model_pipeline)
  --data-path PATH        CSV file or directory containing CSV files
  --random-seed INTEGER   Random seed (default: 42)
  --epochs INTEGER        Number of LightGBM trees (default: 100)
  --output-dir PATH       Output root for data/ and models/
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-name)
            [[ $# -ge 2 ]] || { echo "Missing value for --model-name" >&2; exit 2; }
            MODEL_NAME="$2"
            shift 2
            ;;
        --data-path)
            [[ $# -ge 2 ]] || { echo "Missing value for --data-path" >&2; exit 2; }
            DATA_PATH="$2"
            shift 2
            ;;
        --random-seed)
            [[ $# -ge 2 ]] || { echo "Missing value for --random-seed" >&2; exit 2; }
            RANDOM_SEED="$2"
            shift 2
            ;;
        --epochs)
            [[ $# -ge 2 ]] || { echo "Missing value for --epochs" >&2; exit 2; }
            EPOCHS="$2"
            shift 2
            ;;
        --output-dir)
            [[ $# -ge 2 ]] || { echo "Missing value for --output-dir" >&2; exit 2; }
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -n "${UV_BIN:-}" ]]; then
    UV_COMMAND="${UV_BIN}"
elif command -v uv >/dev/null 2>&1; then
    UV_COMMAND="$(command -v uv)"
else
    UV_WINDOWS_PATH="$(find /mnt/c/Users /c/Users -type f -name uv.exe -path '*astral-sh.uv*' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${UV_WINDOWS_PATH}" ]]; then
        UV_COMMAND="${UV_WINDOWS_PATH}"
    else
        echo "uv is not installed. Install it with: winget install --id=astral-sh.uv -e" >&2
        exit 127
    fi
fi

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo "pyproject.toml was not found in ${PROJECT_ROOT}" >&2
    exit 1
fi

if [[ ! -d "${DATA_PATH}" && ! -f "${DATA_PATH}" ]]; then
    echo "Data path does not exist: ${DATA_PATH}" >&2
    exit 1
fi

if ! [[ "${RANDOM_SEED}" =~ ^[0-9]+$ ]]; then
    echo "--random-seed must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--epochs must be a positive integer" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
if [[ "${DATA_PATH}" == /mnt/c/* ]]; then
    DATA_PATH="$(wslpath -a "${DATA_PATH}")"
elif [[ "${DATA_PATH}" != /* ]]; then
    DATA_PATH="${PROJECT_ROOT}/${DATA_PATH}"
fi
if [[ "${OUTPUT_DIR}" == /mnt/c/* ]]; then
    OUTPUT_DIR="$(wslpath -a "${OUTPUT_DIR}")"
elif [[ "${OUTPUT_DIR}" != /* ]]; then
    OUTPUT_DIR="${PROJECT_ROOT}/${OUTPUT_DIR}"
fi

if [[ "${UV_COMMAND}" == *.exe ]]; then
    UV_COMMAND="${UV_COMMAND}"
fi

"${UV_COMMAND}" sync
"${UV_COMMAND}" run python src/experiments/prepare_datasets.py \
    --model-name "${MODEL_NAME}" \
    --data-path "${DATA_PATH}" \
    --random-seed "${RANDOM_SEED}" \
    --epochs "${EPOCHS}" \
    --output-dir "${OUTPUT_DIR}"
"${UV_COMMAND}" run python src/experiments/validate_pipeline.py \
    --model-name "${MODEL_NAME}" \
    --output-dir "${OUTPUT_DIR}"
