cmake_minimum_required(VERSION 3.10)
project(vtsimnx_solver)

# C++17標準を要求
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# インクルードディレクトリ
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/nlohmann)
include_directories(${CMAKE_SOURCE_DIR}/utils)
include_directories(${CMAKE_SOURCE_DIR}/../archenv/include)
include_directories(${CMAKE_SOURCE_DIR}/../archenv)
include_directories(${CMAKE_SOURCE_DIR}/..)

# ソースファイルの定義
set(PARSER_SOURCES
    parser/sim_constants_parser.cpp
    parser/nodes_parser.cpp
    parser/branches_parser.cpp
)

set(NETWORK_SOURCES
    network/ventilation_network.cpp
    network/thermal_network.cpp
)

set(CORE_SOURCES
    core/pressure_solver.cpp
    core/pressure_solver_ceres.cpp
    core/pressure_solver_fallback.cpp
    core/thermal_solver.cpp
    core/thermal_solver_linear_gs.cpp
)

set(AIRCON_SOURCES
    aircon/aircon_controller.cpp
)

set(ACMODEL_SOURCES
    ../acmodel/acmodel.cpp
    ../acmodel/aircon_constants.cpp
    ../acmodel/aircon_spec.cpp
    ../acmodel/refrigerant_calculator.cpp
    ../acmodel/criepi_model.cpp
    ../acmodel/rac_model.cpp
    ../acmodel/duct_central_model.cpp
    ../acmodel/latent_evaluate_model.cpp
)

set(ARCHENV_SOURCES
    ../archenv/archenv_core.cpp
    ../archenv/archenv_pmv.cpp
    ../archenv/archenv_jis.cpp
    ../archenv/archenv_fungal.cpp
)

set(MAIN_SOURCES
    vtsimnx_solver.cpp
    simulation_runner.cpp
    app/vtsimnx_app.cpp
    output/artifact_io.cpp
)

# すべてのソースファイル
set(ALL_SOURCES
    ${PARSER_SOURCES}
    ${NETWORK_SOURCES}
    ${CORE_SOURCES}
    ${AIRCON_SOURCES}
    ${ACMODEL_SOURCES}
    ${ARCHENV_SOURCES}
    ${MAIN_SOURCES}
)

# 実行ファイルの出力先をトップレベルのbuildディレクトリに設定
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../build)

# 実行ファイルの作成
add_executable(vtsimnx_solver ${ALL_SOURCES})

# nlohmann/jsonの検索（ヘッダーのみ）
# solver/nlohmann/にjson.hppがあることを確認
if(EXISTS "${CMAKE_SOURCE_DIR}/nlohmann/json.hpp")
    set(NLOHMANN_JSON_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/nlohmann)
    message(STATUS "Found nlohmann/json at ${NLOHMANN_JSON_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "nlohmann/json.hpp not found at ${CMAKE_SOURCE_DIR}/nlohmann/json.hpp")
endif()

# Boost Graph Libraryの検索（ヘッダーのみ）
find_package(Boost REQUIRED)

# Ceres Solverの検索
find_package(Ceres REQUIRED)

# リンクライブラリ
target_link_libraries(vtsimnx_solver
    Boost::boost
    ceres
)

# コンパイルオプション
target_compile_options(vtsimnx_solver PRIVATE
    -Wall
    -Wextra
)

# =============================================================================
# C++ unit tests (minimal harness)
# =============================================================================
option(VTSIMNX_BUILD_CPP_TESTS "Build C++ unit tests for solver" ON)
if(VTSIMNX_BUILD_CPP_TESTS)
    enable_testing()
    add_executable(vtsimnx_solver_cpp_tests
        tests_cpp/test_flow_math.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_tests
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_tests PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_tests COMMAND vtsimnx_solver_cpp_tests)
endif()

