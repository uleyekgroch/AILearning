#!/usr/bin/env bash
#
# quantize_model.sh — GGUF 模型量化工具
#
# 用法:
#   ./quantize_model.sh input-f16.gguf output-q4_k_m.gguf q4_k_m
#   ./quantize_model.sh --list          # 列出支持的量化类型
#
# 量化类型 (推荐):
#   q8_0    — INT8  量化，~50% 压缩，精度损失最小
#   q4_k_m  — Q4_K_M 混合量化，~25% 压缩，推荐平衡点
#   q4_k_s  — Q4_K_S 更小体积，速度更快
#   iq4_nl  — IQ4_NL 高质量 4-bit
#   q3_k_m  — Q3_K_M 更高压缩
#   q2_k    — Q2_K  极限压缩（仅边缘设备）
#
# 依赖:
#   llama.cpp 编译后的 quantize 可执行文件
#   如果没有，脚本会尝试自动构建

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/.."

# ── 颜色输出 ──────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()   { echo -e "${RED}[ERR]${NC} $*" >&2; }

# ── 查找 quantize 工具 ──────────────────────────────────────────
find_quantize_tool() {
    local candidates=(
        "${PROJECT_DIR}/build-wsl/bin/quantize"
        "${PROJECT_DIR}/build/bin/quantize"
        "${PROJECT_DIR}/build-wsl/_deps/llama_cpp-build/bin/quantize"
        "${PROJECT_DIR}/build/_deps/llama_cpp-build/bin/quantize"
        "${PROJECT_DIR}/build-wsl/quantize"
        "${PROJECT_DIR}/build/quantize"
    )
    for c in "${candidates[@]}"; do
        if [[ -x "$c" ]]; then
            echo "$c"
            return 0
        fi
    done
    return 1
}

# ── 自动构建 quantize ───────────────────────────────────────────
build_quantize() {
    warn "quantize tool not found. Attempting to build..."
    local build_dir="${PROJECT_DIR}/build-wsl"
    mkdir -p "${build_dir}"
    cd "${build_dir}"
    cmake -DLLAMA_BUILD_EXAMPLES=ON .. 2>/dev/null || {
        err "Failed to configure cmake for quantize build"
        return 1
    }
    cmake --build . --target quantize -j$(nproc 2>/dev/null || echo 4) 2>/dev/null || {
        err "Failed to build quantize tool"
        return 1
    }
    ok "quantize tool built successfully"
}

# ── 列出量化类型 ──────────────────────────────────────────────────
list_types() {
    cat <<'EOF'
支持的量化类型 (llama.cpp):

  类型      说明                          推荐场景
  ─────────────────────────────────────────────────────────────
  q8_0      INT8 量化 (256 原子)           精度优先，~50% 体积
  q6_k      Q6_K 混合量化                   高精度 6-bit
  q5_k_m    Q5_K_M 混合量化                 平衡精度/速度
  q5_k_s    Q5_K_S 更小更快                 速度优先
  q4_k_m    Q4_K_M 混合量化 (默认推荐)      最佳平衡点 ★
  q4_k_s    Q4_K_S 更小更快                 边缘设备
  q4_0      Q4_0 经典 4-bit                 兼容性好
  q4_1      Q4_1 稍高精度                   较老的量化方案
  iq4_nl    IQ4_NL 高质量 4-bit               新方案，质量高
  iq4_xs    IQ4_XS 超小体积                 极限压缩
  q3_k_m    Q3_K_M 混合量化                 高压缩，精度可接受
  q3_k_s    Q3_K_S 更小更快                 极端体积限制
  q2_k      Q2_K 极限压缩                   仅用于测试/边缘

参考: https://github.com/ggml-org/llama.cpp/blob/master/examples/quantize/README.md
EOF
}

# ── 主逻辑 ────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") <input-f16.gguf> <output.gguf> <type>
       $(basename "$0") --list

量化 GGUF 模型以减小体积、提升推理速度。

Examples:
  $(basename "$0") model-f16.gguf model-q4.gguf q4_k_m
  $(basename "$0") --list
EOF
}

main() {
    if [[ $# -eq 1 && "$1" == "--list" ]]; then
        list_types
        exit 0
    fi

    if [[ $# -lt 3 ]]; then
        usage
        exit 1
    fi

    local input="$1"
    local output="$2"
    local qtype="$3"

    if [[ ! -f "$input" ]]; then
        err "Input file not found: $input"
        exit 1
    fi

    local quantize_tool
    quantize_tool=$(find_quantize_tool) || {
        build_quantize
        quantize_tool=$(find_quantize_tool) || {
            err "Could not find or build quantize tool"
            exit 1
        }
    }

    info "Using quantize: ${quantize_tool}"
    info "Input:  ${input} ($(stat -c%s "$input" 2>/dev/null || stat -f%z "$input" 2>/dev/null || echo "?") bytes)"
    info "Output: ${output}"
    info "Type:   ${qtype}"

    "${quantize_tool}" "${input}" "${output}" "${qtype}"

    if [[ -f "$output" ]]; then
        local out_size
        out_size=$(stat -c%s "$output" 2>/dev/null || stat -f%z "$output" 2>/dev/null || echo "?")
        local in_size
        in_size=$(stat -c%s "$input" 2>/dev/null || stat -f%z "$input" 2>/dev/null || echo "1")
        local ratio
        ratio=$(awk "BEGIN {printf \"%.1f\", 100 * ${out_size} / ${in_size}}")
        ok "Quantization complete: ${output} (${out_size} bytes, ${ratio}% of original)"
    else
        err "Quantization failed — output file not created"
        exit 1
    fi
}

main "$@"
