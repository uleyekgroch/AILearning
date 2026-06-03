#!/bin/bash
# ══════════════════════════════════════════════════════════════
#  ai-learning-cpp WSL 一键构建脚本 (CPU / GCC)
# ══════════════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ── 1. 安装依赖 ──────────────────────────────────────────────
install_deps() {
    info "检查并安装构建依赖..."
    sudo apt-get update -qq

    local pkgs=()
    for cmd in g++ cmake git make; do
        if ! command -v $cmd &>/dev/null; then
            pkgs+=( $cmd )
        fi
    done

    if (( ${#pkgs[@]} )); then
        info "安装: ${pkgs[*]}"
        sudo apt-get install -y -qq "${pkgs[@]}"
    fi

    # 验证 GCC 版本 (需要 C++20 支持 => GCC >= 10)
    local gcc_ver
    gcc_ver=$(g++ -dumpversion | cut -d. -f1)
    if (( gcc_ver < 10 )); then
        warn "GCC ${gcc_ver} 不支持 C++20，安装 g++-12..."
        sudo apt-get install -y -qq g++-12
        sudo update-alternatives --set g++ /usr/bin/g++-12 2>/dev/null || true
    fi

    ok "g++ $(g++ -dumpversion), cmake $(cmake --version | head -1 | awk '{print $3}')"
}

# ── 2. 构建 ──────────────────────────────────────────────────
build() {
    local src_dir="$1"
    local build_dir="${src_dir}/build-wsl"

    info "构建目录: ${build_dir}"
    mkdir -p "${build_dir}"
    cd "${build_dir}"

    info "CMake 配置..."
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER=gcc \
        -DCMAKE_CXX_COMPILER=g++ \
        -G "Unix Makefiles" \
        2>&1 | tail -5

    info "编译 ($(nproc) 线程)..."
    cmake --build . --parallel "$(nproc)" 2>&1 | tail -20

    ok "编译完成"
}

# ── 3. 运行测试 ──────────────────────────────────────────────
run_tests() {
    local src_dir="$1"
    local build_dir="${src_dir}/build-wsl"

    info "运行测试..."
    cd "${build_dir}"
    ctest --output-on-failure --parallel "$(nproc)" 2>&1 | tail -30

    ok "测试完成"
}

# ── 4. 运行基准 ──────────────────────────────────────────────
run_benchmark() {
    local src_dir="$1"
    local build_dir="${src_dir}/build-wsl"

    info "运行基准测试..."
    "${build_dir}/ai_learning_benchmark" 2>&1 | tail -30
}

# ── main ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ai-learning-cpp  WSL CPU Build"
echo "════════════════════════════════════════════════════════════"
echo ""

install_deps
build "${SCRIPT_DIR}"

echo ""
read -p "运行测试? [Y/n] " -n1 -r ans
echo ""
[[ "${ans,,}" != "n" ]] && run_tests "${SCRIPT_DIR}"

echo ""
read -p "运行基准? [Y/n] " -n1 -r ans
echo ""
[[ "${ans,,}" != "n" ]] && run_benchmark "${SCRIPT_DIR}"

echo ""
ok "全部完成! 🎉"
