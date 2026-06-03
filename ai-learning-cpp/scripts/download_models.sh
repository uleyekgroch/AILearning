#!/usr/bin/env bash
#
# download_models.sh — 自动下载推荐的 llama.cpp 模型
#
# 用法:
#   ./download_models.sh                    # 下载所有推荐模型到 ../models/
#   ./download_models.sh --dir /custom/path # 指定下载目录
#   ./download_models.sh --model embedding  # 仅下载 embedding 模型
#   ./download_models.sh --model llm        # 仅下载对话模型
#
# 支持的镜像（自动回退）:
#   1. hf-mirror.com      (国内加速)
#   2. huggingface.co     (官方)
#
# 模型清单:
#   • bge-small-zh-v1.5-q4_k_m.gguf    (~15 MB)  — 中文 Embedding
#   • qwen2.5-3b-instruct-q4_k_m.gguf  (~2.0 GB) — 中文对话/推理
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MODEL_DIR="${SCRIPT_DIR}/../models"
MODEL_DIR="${DEFAULT_MODEL_DIR}"

# ── 颜色输出 ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── 模型配置 ──────────────────────────────────────────────────────
declare -A MODEL_URLS
declare -A MODEL_SIZES

# bge-small-zh-v1.5 Q4_K_M (~15MB)
MODEL_URLS[embedding]="https://hf-mirror.com/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-q4_k_m.gguf"
MODEL_SIZES[embedding]=15728640  # 15 MB approx

# Qwen2.5-3B-Instruct Q4_K_M (~2GB)
MODEL_URLS[llm]="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
MODEL_SIZES[llm]=2147483648  # 2 GB approx

# 备用镜像
MIRRORS=(
    "https://hf-mirror.com"
    "https://huggingface.co"
)

# ── 下载函数 ──────────────────────────────────────────────────────
download_with_resume() {
    local url="$1"
    local output="$2"
    local desc="$3"

    info "Downloading ${desc}..."
    info "  URL: ${url}"
    info "  Dest: ${output}"

    # 检查是否已存在且完整
    if [[ -f "${output}" ]]; then
        local existing_size
        existing_size=$(stat -f%z "${output}" 2>/dev/null || stat -c%s "${output}" 2>/dev/null || echo 0)
        if [[ ${existing_size} -gt 1000000 ]]; then
            warn "File already exists (${existing_size} bytes). Skipping."
            warn "Use rm '${output}' to force re-download."
            return 0
        fi
    fi

    # 优先使用 wget（带进度条），fallback 到 curl
    if command -v wget &>/dev/null; then
        wget --continue --progress=bar:force --show-progress \
             --timeout=60 --tries=3 \
             -O "${output}" "${url}" 2>&1 || return 1
    elif command -v curl &>/dev/null; then
        curl -L -C - --progress-bar \
             --max-time 300 --retry 3 \
             -o "${output}" "${url}" 2>&1 || return 1
    else
        err "Neither wget nor curl is installed. Please install one."
        return 1
    fi

    # 验证文件大小（至少 1MB）
    local downloaded_size
    downloaded_size=$(stat -f%z "${output}" 2>/dev/null || stat -c%s "${output}" 2>/dev/null || echo 0)
    if [[ ${downloaded_size} -lt 1000000 ]]; then
        err "Downloaded file is too small (${downloaded_size} bytes). May be an error page."
        rm -f "${output}"
        return 1
    fi

    ok "Downloaded: ${output} (${downloaded_size} bytes)"
    return 0
}

# 尝试多个镜像
download_with_fallback() {
    local primary_url="$1"
    local output="$2"
    local desc="$3"

    # 尝试主 URL
    if download_with_resume "${primary_url}" "${output}" "${desc}"; then
        return 0
    fi

    # 尝试备用镜像（替换域名）
    for mirror in "${MIRRORS[@]}"; do
        if [[ "${primary_url}" == *"hf-mirror.com"* && "${mirror}" == *"huggingface.co"* ]]; then
            local alt_url="${primary_url/hf-mirror.com/huggingface.co}"
        elif [[ "${primary_url}" == *"huggingface.co"* && "${mirror}" == *"hf-mirror.com"* ]]; then
            local alt_url="${primary_url/huggingface.co/hf-mirror.com}"
        else
            continue
        fi

        warn "Primary mirror failed. Trying fallback: ${mirror}"
        if download_with_resume "${alt_url}" "${output}" "${desc} (fallback)"; then
            return 0
        fi
    done

    err "All mirrors failed for ${desc}"
    return 1
}

# ── 主逻辑 ────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

自动下载推荐的 llama.cpp 预训练模型。

Options:
  --dir PATH       指定模型下载目录 (默认: ../models)
  --model TYPE     仅下载指定类型: embedding | llm | all (默认: all)
  --help           显示此帮助

Examples:
  $(basename "$0")                           # 下载所有模型
  $(basename "$0") --model embedding         # 仅下载 bge-small-zh
  $(basename "$0") --dir ~/my-models       # 下载到自定义目录
EOF
}

main() {
    local model_filter="all"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dir)
                MODEL_DIR="$2"
                shift 2
                ;;
            --model)
                model_filter="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                err "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done

    mkdir -p "${MODEL_DIR}"
    info "Model directory: ${MODEL_DIR}"
    echo ""

    local failed=0

    # ── Embedding 模型 ────────────────────────────────────────────
    if [[ "${model_filter}" == "all" || "${model_filter}" == "embedding" ]]; then
        info "========================================"
        info "Model: bge-small-zh-v1.5 Q4_K_M"
        info "Purpose: Chinese text embedding (~15MB)"
        info "========================================"

        local embedding_path="${MODEL_DIR}/bge-small-zh-v1.5-q4_k_m.gguf"
        if ! download_with_fallback "${MODEL_URLS[embedding]}" "${embedding_path}" "bge-small-zh"; then
            ((failed++)) || true
        fi
        echo ""
    fi

    # ── LLM 模型 ──────────────────────────────────────────────────
    if [[ "${model_filter}" == "all" || "${model_filter}" == "llm" ]]; then
        info "========================================"
        info "Model: Qwen2.5-3B-Instruct Q4_K_M"
        info "Purpose: Chinese dialogue & reasoning (~2GB)"
        info "========================================"
        warn "This is a large file (~2GB). Download may take several minutes."

        local llm_path="${MODEL_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf"
        if ! download_with_fallback "${MODEL_URLS[llm]}" "${llm_path}" "Qwen2.5-3B"; then
            ((failed++)) || true
        fi
        echo ""
    fi

    # ── 摘要 ──────────────────────────────────────────────────────
    echo ""
    info "========================================"
    info "Download Summary"
    info "========================================"

    ls -lh "${MODEL_DIR}"/*.gguf 2>/dev/null || true

    echo ""
    if [[ ${failed} -eq 0 ]]; then
        ok "All requested models downloaded successfully!"
        echo ""
        info "Next steps:"
        echo "  ./verify_llama_cpp ${MODEL_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf"
        echo "  ./ai_learning_server --llm-model ${MODEL_DIR}/qwen2.5-3b-instruct-q4_k_m.gguf"
        exit 0
    else
        err "${failed} download(s) failed."
        err "Check your network connection or try manually from:"
        err "  https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf"
        err "  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF"
        exit 1
    fi
}

main "$@"
