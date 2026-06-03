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
