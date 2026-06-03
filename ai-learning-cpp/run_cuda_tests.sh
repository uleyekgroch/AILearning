#!/bin/bash
# run_cuda_tests.sh - Run CUDA build tests + benchmark
set -e

export PATH="/usr/local/cuda-12.6/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH:-}"

BUILD_DIR="/mnt/d/mayAi/AILearning_v0527/ai-learning-cpp/build-cuda"
cd "${BUILD_DIR}"

echo "=== Running Tests ==="
./ai_learning_tests 2>&1 || true

echo ""
echo "=== Benchmark ==="
./ai_learning_benchmark 2>&1 || true

echo ""
echo "=== DONE ==="
