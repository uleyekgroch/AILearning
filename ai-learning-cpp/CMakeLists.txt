cmake_minimum_required(VERSION 3.22)
project(ai_learning VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── CUDA 检测 ────────────────────────────────────────────────────
# 检测 CUDA Toolkit 是否可用（可选，不可用则纯 CPU 构建）
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_FOUND TRUE)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CMake 3.22 需要显式找 CUDA 包以获得 CUDA::cudart 等目标
    find_package(CUDAToolkit REQUIRED)

    # 允许在 CUDA 代码中使用 __host__/__device__ lambda（cuda_fp16_utils.cuh 需要）
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

    # 设置 CUDA 架构（支持 RTX 4060 Laptop 的 sm_89）
    # 也可设为 "native" 自动检测，但 CMake 3.22 可能不支持
    set(CMAKE_CUDA_ARCHITECTURES "89")

    message(STATUS "✅ CUDA enabled: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_FOUND FALSE)
    message(STATUS "⚠️  CUDA not found — building CPU-only")
endif()

# ── 编译选项 ────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(
        /W4                 # 高警告级别
        /WX                 # 警告视为错误
        /permissive-        # 严格标准一致性
        /Zc:__cplusplus     # 正确报告 __cplusplus 值
        /utf-8              # 源码 UTF-8
    )
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(/Od /Zi)
    else()
        add_compile_options(/O2 /DNDEBUG)
    endif()
else()
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>
                        $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                        $<$<COMPILE_LANGUAGE:CXX>:-Werror>
                        $<$<COMPILE_LANGUAGE:CXX>:-std=c++20>)
endif()

# ── llama.cpp 可选集成 ────────────────────────────────────────────
option(AI_LEARNING_WITH_LLAMA_CPP
    "Enable llama.cpp integration for pre-trained embeddings" OFF)

# ── 外部依赖 ─────────────────────────────────────────────────────
include(FetchContent)

# Catch2 测试框架
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
)

# nlohmann/json — JSON 序列化/反序列化
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)

# Asio — Crow 的异步 I/O 依赖（standalone 模式）
# 使用 GitHub tag archive zip 下载（比 git clone 快得多，避免网络超时）
# 解压后目录结构：asio-asio-1-30-2/asio/include/asio.hpp
FetchContent_Declare(
    asio
    URL https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-30-2.zip
)
FetchContent_MakeAvailable(asio)

# 设置 ASIO_INCLUDE_DIR，让 Crow 的 Findasio.cmake 能找到 asio 头文件
if(asio_SOURCE_DIR)
    set(ASIO_INCLUDE_DIR "${asio_SOURCE_DIR}/asio/include" CACHE PATH "asio include directory" FORCE)
endif()

# Crow — header-only HTTP 框架（Flask-like）
FetchContent_Declare(
    Crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG        v1.2.0
)

# llama.cpp — 可选预训练模型后端
if(AI_LEARNING_WITH_LLAMA_CPP)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    # 自动传递 CUDA 设置给 llama.cpp
    if(CUDA_FOUND)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
        message(STATUS "✅ llama.cpp CUDA backend enabled")
    endif()
    FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        b4419
    )
    FetchContent_MakeAvailable(Catch2 nlohmann_json Crow llama_cpp)
else()
    FetchContent_MakeAvailable(Catch2 nlohmann_json Crow)
endif()

# ── 项目库 ──────────────────────────────────────────────────────
file(GLOB_RECURSE AI_LEARNING_SOURCES "src/*.cpp")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "tensor_ops_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "stdp_learning_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "activation_spread_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "analogical_transfer_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "flash_attention_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "knowledge_graph_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "active_experimenter_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "embedding_trainer_cuda_stub\\.cpp$")
list(FILTER AI_LEARNING_SOURCES EXCLUDE REGEX "src/server/")

if(CUDA_FOUND)
    # CUDA 源文件单独收集，避免 .cu 被 C++ 编译器处理
    file(GLOB_RECURSE AI_LEARNING_CUDA_SOURCES "src/*.cu")
    # 创建 CUDA 静态库（包含 .cu → .o）
    add_library(ai_learning_cuda STATIC ${AI_LEARNING_CUDA_SOURCES})
    target_include_directories(ai_learning_cuda PUBLIC include)
    set_target_properties(ai_learning_cuda PROPERTIES
        CUDA_ARCHITECTURES "89"    # RTX 4060 Ada = sm_89 (Ada Lovelace)
        POSITION_INDEPENDENT_CODE ON
    )
    target_link_libraries(ai_learning_cuda
        PUBLIC CUDA::cudart CUDA::cublas)
    target_compile_options(ai_learning_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
else()
    # 无 CUDA 时，添加 stub 实现（cuda_available() 返回 false）
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/core/tensor_ops_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/learning/stdp_learning_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/reasoning/activation_spread_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/learning/analogical_transfer_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/core/flash_attention_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/domain/knowledge/knowledge_graph_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/learning/active_experimenter_cuda_stub.cpp")
    list(APPEND AI_LEARNING_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/learning/embedding_trainer_cuda_stub.cpp")
endif()

add_library(ai_learning STATIC ${AI_LEARNING_SOURCES})
target_include_directories(ai_learning PUBLIC include)
target_compile_features(ai_learning PRIVATE cxx_std_20)
# ai_learning 内部源码依赖 nlohmann/json 和 asio
target_link_libraries(ai_learning PUBLIC nlohmann_json::nlohmann_json)
if(asio_SOURCE_DIR)
    target_include_directories(ai_learning PUBLIC "${asio_SOURCE_DIR}/asio/include")
endif()
target_compile_definitions(ai_learning PUBLIC ASIO_STANDALONE)
# Windows: asio requires _WIN32_WINNT definition
if(WIN32)
    target_compile_definitions(ai_learning PUBLIC _WIN32_WINNT=0x0601)
endif()

if(AI_LEARNING_WITH_LLAMA_CPP)
    target_compile_definitions(ai_learning PUBLIC AI_LEARNING_WITH_LLAMA_CPP)
    target_link_libraries(ai_learning PUBLIC llama)
endif()

if(CUDA_FOUND)
    target_link_libraries(ai_learning PUBLIC ai_learning_cuda)
endif()

# ── 测试 ────────────────────────────────────────────────────────
enable_testing()

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
# 分离服务器路由测试（需要 Crow 依赖）
file(GLOB_RECURSE SERVER_TEST_SOURCES "tests/server/*.cpp")
list(REMOVE_ITEM TEST_SOURCES ${SERVER_TEST_SOURCES})

# 主测试可执行文件（不含 Crow 依赖）
add_executable(ai_learning_tests ${TEST_SOURCES})
target_link_libraries(ai_learning_tests
    PRIVATE ai_learning Catch2::Catch2WithMain
)
target_include_directories(ai_learning_tests PRIVATE include)

# 服务器路由测试可执行文件（需要 Crow + server 头文件 + server 源码）
if(SERVER_TEST_SOURCES)
    # 收集 server 实现源码（路由注册 + 事件适配器）
    file(GLOB SERVER_IMPL_SOURCES "src/server/*.cpp")
    add_executable(ai_learning_server_tests ${SERVER_TEST_SOURCES} ${SERVER_IMPL_SOURCES})
    target_link_libraries(ai_learning_server_tests
        PRIVATE ai_learning Crow::Crow nlohmann_json::nlohmann_json Catch2::Catch2WithMain
    )
    target_include_directories(ai_learning_server_tests PRIVATE include src/server)
    target_compile_definitions(ai_learning_server_tests PRIVATE ASIO_STANDALONE)
    target_compile_features(ai_learning_server_tests PRIVATE cxx_std_20)
    if(WIN32)
        target_link_libraries(ai_learning_server_tests PRIVATE ws2_32 wsock32)
    endif()
endif()

include(CTest)
list(APPEND CMAKE_PREFIX_PATH "${Catch2_DIR}")
include(Catch)
# Skip catch_discover_tests on MinGW (DLL not found in build env).
# Run manually: PATH="/d/msys64/ucrt64/bin:$PATH" ./ai_learning_tests.exe
# catch_discover_tests(ai_learning_tests)
# Server tests: run manually:
# PATH="/d/msys64/ucrt64/bin:$PATH" ./ai_learning_server_tests.exe
# if(TARGET ai_learning_server_tests)
#     catch_discover_tests(ai_learning_server_tests)
# endif()

# ── 主程序 ──────────────────────────────────────────────────────
add_executable(ai_learning_main main.cpp)
target_link_libraries(ai_learning_main PRIVATE ai_learning)

# ── 性能基准 ──────────────────────────────────────────────────────
add_executable(ai_learning_benchmark benchmark.cpp)
target_link_libraries(ai_learning_benchmark PRIVATE ai_learning)

add_executable(ai_learning_benchmark_engines benchmark_engines.cpp)
target_link_libraries(ai_learning_benchmark_engines PRIVATE ai_learning)

# ── 调试程序 ────────────────────────────────────────────────────
add_executable(debug_main debug_main.cpp)
target_link_libraries(debug_main PRIVATE ai_learning)
target_include_directories(debug_main PRIVATE include)

# ── REST 服务 ────────────────────────────────────────────────────
add_executable(ai_learning_server
    server_main.cpp
    src/server/server.cpp
    src/server/event_adapter.cpp
    src/server/core_routes.cpp
    src/server/advanced_routes.cpp
    src/server/society_routes.cpp
    src/server/chat_routes.cpp
    src/server/runtime_routes.cpp
    src/server/goals_routes.cpp
)
target_link_libraries(ai_learning_server
    PRIVATE ai_learning Crow::Crow nlohmann_json::nlohmann_json
)
target_include_directories(ai_learning_server PRIVATE include)
# Crow 依赖 ASIO，需要 ASIO_STANDALONE
target_compile_definitions(ai_learning_server PRIVATE ASIO_STANDALONE)
# Windows 上 asio 需要 winsock 库
if(WIN32)
    target_link_libraries(ai_learning_server PRIVATE ws2_32 wsock32)
endif()

# ── 语料学习 ────────────────────────────────────────────────────
add_executable(ai_learning_corpus corpus_main.cpp)
target_link_libraries(ai_learning_corpus PRIVATE ai_learning nlohmann_json::nlohmann_json)
target_include_directories(ai_learning_corpus PRIVATE include)

# ── llama.cpp 三阶段验证 ─────────────────────────────────────────
if(AI_LEARNING_WITH_LLAMA_CPP)
    add_executable(verify_llama_cpp verify_llama_cpp.cpp)
    target_link_libraries(verify_llama_cpp PRIVATE ai_learning)
    target_include_directories(verify_llama_cpp PRIVATE include)

    add_executable(compare_embeddings compare_embeddings.cpp)
    target_link_libraries(compare_embeddings PRIVATE ai_learning)
    target_include_directories(compare_embeddings PRIVATE include)
endif()
