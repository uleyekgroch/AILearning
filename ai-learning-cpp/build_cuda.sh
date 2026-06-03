#!/bin/bash
# build_cuda.sh - CUDA build (clean PATH to avoid Windows spaces/parens)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-cuda"

# Override PATH completely — only Linux + CUDA paths
export PATH="/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64"

echo "=== Phase 1: CMake Configure ==="
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.6/bin/nvcc \
    -G "Unix Makefiles"

echo ""
echo "=== Phase 2: Compile ==="
cmake --build . --parallel "$(nproc)"
echo ""
echo "=== Build Complete ==="
