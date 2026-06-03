/**
 * @file test_tensor_ops.cpp
 * @brief 张量运算单元测试
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "ai_learning/core/tensor_ops.hpp"

using namespace ai_learning::core;
using Catch::Matchers::WithinAbs;

TEST_CASE("TensorOps: 向量加法") {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {4.0f, 5.0f, 6.0f};
    auto result = tensor_add(a, b);
    REQUIRE(result == Tensor{5.0f, 7.0f, 9.0f});
}

TEST_CASE("TensorOps: 向量减法") {
    Tensor a = {5.0f, 3.0f, 1.0f};
    Tensor b = {1.0f, 2.0f, 3.0f};
    auto result = tensor_sub(a, b);
    REQUIRE(result == Tensor{4.0f, 1.0f, -2.0f});
}

TEST_CASE("TensorOps: 标量乘法") {
    Tensor a = {1.0f, 2.0f, 3.0f};
    auto result = tensor_scale(a, 2.0f);
    REQUIRE(result == Tensor{2.0f, 4.0f, 6.0f});
}

TEST_CASE("TensorOps: ReLU") {
    Tensor a = {-1.0f, 0.0f, 1.0f, -0.5f};
    auto result = tensor_relu(a);
    REQUIRE(result == Tensor{0.0f, 0.0f, 1.0f, 0.0f});
}

TEST_CASE("TensorOps: MSE") {
    Tensor a = {1.0f, 2.0f, 3.0f};
    Tensor b = {1.0f, 2.0f, 3.0f};
    REQUIRE_THAT(tensor_mse(a, b), WithinAbs(0.0, 1e-6));

    Tensor c = {2.0f, 3.0f, 4.0f};
    REQUIRE_THAT(tensor_mse(a, c), WithinAbs(1.0, 1e-6));
}

TEST_CASE("TensorOps: 余弦相似度") {
    Tensor a = {1.0f, 0.0f, 0.0f};
    Tensor b = {1.0f, 0.0f, 0.0f};
    REQUIRE_THAT(cosine_similarity(a, b), WithinAbs(1.0, 1e-6));

    Tensor c = {0.0f, 1.0f, 0.0f};
    REQUIRE_THAT(cosine_similarity(a, c), WithinAbs(0.0, 1e-6));

    Tensor d = {1.0f, 1.0f, 0.0f};
    REQUIRE_THAT(cosine_similarity(a, d), WithinAbs(0.707, 0.01));
}

TEST_CASE("TensorOps: 矩阵向量乘法") {
    // 2×3 矩阵
    std::vector<float> mat = {1, 0, 0,  0, 1, 0};
    Tensor vec = {3.0f, 4.0f, 0.0f};
    auto result = mat_vec(mat, 2, 3, vec);
    REQUIRE(result == Tensor{3.0f, 4.0f});
}

TEST_CASE("TensorOps: 外积更新") {
    std::vector<float> mat(4, 0.0f);
    Tensor a = {1.0f, 2.0f};
    Tensor b = {3.0f, 4.0f};
    mat_add_outer(mat, 1.0f, a, b);

    // outer(a, b) = [[3, 4], [6, 8]]
    REQUIRE(mat[0] == 3.0f);
    REQUIRE(mat[1] == 4.0f);
    REQUIRE(mat[2] == 6.0f);
    REQUIRE(mat[3] == 8.0f);
}

TEST_CASE("TensorOps: clamp") {
    Tensor a = {-5.0f, -1.0f, 0.0f, 1.0f, 5.0f};
    auto result = tensor_clamp(a, -3.0f, 3.0f);
    REQUIRE(result == Tensor{-3.0f, -1.0f, 0.0f, 1.0f, 3.0f});
}

// ── Matrix 类型安全测试 ────────────────────────────────────────

TEST_CASE("Matrix: 构造和元素访问") {
    Matrix m(2, 3);
    REQUIRE(m.rows == 2);
    REQUIRE(m.cols == 3);
    REQUIRE(m.size() == 6);

    m(0, 0) = 1.0f;
    m(0, 1) = 2.0f;
    m(0, 2) = 3.0f;
    m(1, 0) = 4.0f;
    m(1, 1) = 5.0f;
    m(1, 2) = 6.0f;

    REQUIRE(m(0, 0) == 1.0f);
    REQUIRE(m(1, 2) == 6.0f);
    REQUIRE(m.data[0] == 1.0f);  // row-major flat
    REQUIRE(m.data[5] == 6.0f);
}

TEST_CASE("Matrix: mat_vec 类型安全重载") {
    // 2×3 矩阵: [[1,2,3], [4,5,6]]
    Matrix mat(2, 3);
    mat(0, 0) = 1.0f; mat(0, 1) = 2.0f; mat(0, 2) = 3.0f;
    mat(1, 0) = 4.0f; mat(1, 1) = 5.0f; mat(1, 2) = 6.0f;

    Tensor vec = {1.0f, 0.0f, 0.0f};
    auto result = mat_vec(mat, vec);

    REQUIRE(result.size() == 2);
    REQUIRE(result[0] == 1.0f);
    REQUIRE(result[1] == 4.0f);
}

TEST_CASE("Matrix: mat_vec_bias 类型安全重载") {
    // 2×3 矩阵: [[1,0,0], [0,1,0]]
    Matrix mat(2, 3);
    mat(0, 0) = 1.0f; mat(0, 1) = 0.0f; mat(0, 2) = 0.0f;
    mat(1, 0) = 0.0f; mat(1, 1) = 1.0f; mat(1, 2) = 0.0f;

    Tensor vec = {3.0f, 4.0f, 0.0f};
    Tensor bias = {10.0f, 20.0f};
    auto result = mat_vec_bias(mat, vec, bias);

    REQUIRE(result.size() == 2);
    REQUIRE(result[0] == 13.0f);  // 1*3 + 10
    REQUIRE(result[1] == 24.0f);  // 1*4 + 20
}

TEST_CASE("Matrix: 空矩阵") {
    Matrix m;
    REQUIRE(m.rows == 0);
    REQUIRE(m.cols == 0);
    REQUIRE(m.empty());
}
