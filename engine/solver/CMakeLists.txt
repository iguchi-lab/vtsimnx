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
    network/contaminant_network.cpp
    network/contaminant_network_outputs.cpp
    network/ventilation_network.cpp
    network/ventilation_network_outputs.cpp
    network/thermal_network.cpp
    network/thermal_network_outputs.cpp
    network/humidity_network.cpp
    network/humidity_network_outputs.cpp
)

set(CORE_SOURCES
    core/ventilation/pressure_solver.cpp
    core/ventilation/pressure_solver_ceres.cpp
    core/ventilation/pressure_solver_fallback.cpp
    core/ventilation/pressure_constraints.cpp
    core/thermal/thermal_solver.cpp
    core/thermal/thermal_solver_linear_direct.cpp
    core/thermal/thermal_direct_internal.cpp
    core/thermal/thermal_direct_topology.cpp
    core/thermal/thermal_direct_rhs.cpp
    core/thermal/thermal_direct_build.cpp
    core/thermal/thermal_direct_solve.cpp
    core/thermal/thermal_direct_postprocess.cpp
    core/humidity/humidity_coupling.cpp
    core/humidity/humidity_solver.cpp
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
    simulation_runner_helpers.cpp
    transport/concentration_solver.cpp
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
    # 1) flow math
    add_executable(vtsimnx_solver_cpp_test_flow_math
        tests_cpp/test_flow_math.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_flow_math
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_flow_math PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_flow_math COMMAND vtsimnx_solver_cpp_test_flow_math)

    # 1b) fan jacobian sweep (analytic vs numeric over many parameter sets)
    add_executable(vtsimnx_solver_cpp_test_fan_jacobian_sweep
        tests_cpp/test_fan_jacobian_sweep.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_fan_jacobian_sweep
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_fan_jacobian_sweep PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_fan_jacobian_sweep COMMAND vtsimnx_solver_cpp_test_fan_jacobian_sweep)

    # 2) parser (thermal branches / response_conduction)
    add_executable(vtsimnx_solver_cpp_test_parser_thermal_branches
        tests_cpp/test_parser_thermal_branches.cpp
        parser/branches_parser.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_parser_thermal_branches
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_parser_thermal_branches PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_parser_thermal_branches COMMAND vtsimnx_solver_cpp_test_parser_thermal_branches)

    # 2b) parser (branches validation / enable / duplicate keys)
    add_executable(vtsimnx_solver_cpp_test_parser_branches_validation
        tests_cpp/test_parser_branches_validation.cpp
        parser/branches_parser.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_parser_branches_validation
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_parser_branches_validation PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_parser_branches_validation COMMAND vtsimnx_solver_cpp_test_parser_branches_validation)

    # 3) parser (nodes.mode backward-compat: number -> string)
    add_executable(vtsimnx_solver_cpp_test_parser_nodes_mode
        tests_cpp/test_parser_nodes_mode.cpp
        parser/nodes_parser.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_parser_nodes_mode
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_parser_nodes_mode PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_parser_nodes_mode COMMAND vtsimnx_solver_cpp_test_parser_nodes_mode)

    # 3b) parser (nodes required fields)
    add_executable(vtsimnx_solver_cpp_test_parser_nodes_required
        tests_cpp/test_parser_nodes_required.cpp
        parser/nodes_parser.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_parser_nodes_required
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_parser_nodes_required PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_parser_nodes_required COMMAND vtsimnx_solver_cpp_test_parser_nodes_required)

    # 4) thermal linear utils (hash/pattern/row index map)
    add_executable(vtsimnx_solver_cpp_test_thermal_linear_utils
        tests_cpp/test_thermal_linear_utils.cpp
    )
    target_compile_options(vtsimnx_solver_cpp_test_thermal_linear_utils PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_thermal_linear_utils COMMAND vtsimnx_solver_cpp_test_thermal_linear_utils)

    # 5) thermal direct cache behavior (coeffSig -> rhsOnly/fullBuild switching)
    add_executable(vtsimnx_solver_cpp_test_thermal_direct_cache
        tests_cpp/test_thermal_direct_cache.cpp
        core/thermal/thermal_solver.cpp
        core/thermal/thermal_solver_linear_direct.cpp
        core/thermal/thermal_direct_internal.cpp
        core/thermal/thermal_direct_topology.cpp
        core/thermal/thermal_direct_rhs.cpp
        core/thermal/thermal_direct_build.cpp
        core/thermal/thermal_direct_solve.cpp
        core/thermal/thermal_direct_postprocess.cpp
        network/thermal_network.cpp
        network/thermal_network_outputs.cpp
        network/humidity_network.cpp
        network/humidity_network_outputs.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_thermal_direct_cache
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_thermal_direct_cache PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_thermal_direct_cache COMMAND vtsimnx_solver_cpp_test_thermal_direct_cache)

    # 5b) thermal advection sign: negative flow should keep heat-balance sign consistency
    add_executable(vtsimnx_solver_cpp_test_thermal_advection_sign
        tests_cpp/test_thermal_advection_sign.cpp
        core/thermal/thermal_solver.cpp
        core/thermal/thermal_solver_linear_direct.cpp
        core/thermal/thermal_direct_internal.cpp
        core/thermal/thermal_direct_topology.cpp
        core/thermal/thermal_direct_rhs.cpp
        core/thermal/thermal_direct_build.cpp
        core/thermal/thermal_direct_solve.cpp
        core/thermal/thermal_direct_postprocess.cpp
        network/thermal_network.cpp
        network/thermal_network_outputs.cpp
        network/humidity_network.cpp
        network/humidity_network_outputs.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_thermal_advection_sign
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_thermal_advection_sign PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_thermal_advection_sign COMMAND vtsimnx_solver_cpp_test_thermal_advection_sign)

    # 6) pressure constraints (Evaluate residual/jacobian without running solver)
    add_executable(vtsimnx_solver_cpp_test_pressure_constraints
        tests_cpp/test_pressure_constraints.cpp
        core/ventilation/pressure_constraints.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_pressure_constraints
        Boost::boost
        ceres
    )
    target_compile_options(vtsimnx_solver_cpp_test_pressure_constraints PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_pressure_constraints COMMAND vtsimnx_solver_cpp_test_pressure_constraints)

    # 6b) ventilation network: parallel branches should keep per-branch flow_rate (no aggregation copy)
    add_executable(vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates
        tests_cpp/test_vent_parallel_branch_flow_rates.cpp
        network/ventilation_network.cpp
        network/ventilation_network_outputs.cpp
        core/ventilation/pressure_solver.cpp
        core/ventilation/pressure_solver_ceres.cpp
        core/ventilation/pressure_solver_fallback.cpp
        core/ventilation/pressure_constraints.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates
        Boost::boost
        ceres
    )
    target_compile_options(vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates
             COMMAND vtsimnx_solver_cpp_test_vent_parallel_branch_flow_rates)

    # 7) network: buildFromData idempotency (no accumulation on rebuild)
    add_executable(vtsimnx_solver_cpp_test_network_rebuild
        tests_cpp/test_network_rebuild.cpp
        network/ventilation_network.cpp
        network/ventilation_network_outputs.cpp
        network/thermal_network.cpp
        network/thermal_network_outputs.cpp
        network/humidity_network.cpp
        network/humidity_network_outputs.cpp
        core/ventilation/pressure_solver.cpp
        core/ventilation/pressure_solver_ceres.cpp
        core/ventilation/pressure_solver_fallback.cpp
        core/ventilation/pressure_constraints.cpp
        core/thermal/thermal_solver.cpp
        core/thermal/thermal_solver_linear_direct.cpp
        core/thermal/thermal_direct_internal.cpp
        core/thermal/thermal_direct_topology.cpp
        core/thermal/thermal_direct_rhs.cpp
        core/thermal/thermal_direct_build.cpp
        core/thermal/thermal_direct_solve.cpp
        core/thermal/thermal_direct_postprocess.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_network_rebuild
        Boost::boost
        ceres
    )
    target_compile_options(vtsimnx_solver_cpp_test_network_rebuild PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_network_rebuild COMMAND vtsimnx_solver_cpp_test_network_rebuild)

    # 7b) transport: humidity(x) + concentration(c) 1-step update
    add_executable(vtsimnx_solver_cpp_test_transport_humidity_concentration
        tests_cpp/test_transport_humidity_concentration.cpp
        core/humidity/humidity_solver.cpp
        core/humidity/humidity_coupling.cpp
        transport/concentration_solver.cpp
        network/contaminant_network.cpp
        network/ventilation_network.cpp
        network/ventilation_network_outputs.cpp
        network/thermal_network.cpp
        network/thermal_network_outputs.cpp
        network/humidity_network.cpp
        network/humidity_network_outputs.cpp
        core/ventilation/pressure_solver.cpp
        core/ventilation/pressure_solver_ceres.cpp
        core/ventilation/pressure_solver_fallback.cpp
        core/ventilation/pressure_constraints.cpp
        core/thermal/thermal_solver.cpp
        core/thermal/thermal_solver_linear_direct.cpp
        core/thermal/thermal_direct_internal.cpp
        core/thermal/thermal_direct_topology.cpp
        core/thermal/thermal_direct_rhs.cpp
        core/thermal/thermal_direct_build.cpp
        core/thermal/thermal_direct_solve.cpp
        core/thermal/thermal_direct_postprocess.cpp
    )
    target_link_libraries(vtsimnx_solver_cpp_test_transport_humidity_concentration
        Boost::boost
        ceres
    )
    target_compile_options(vtsimnx_solver_cpp_test_transport_humidity_concentration PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_transport_humidity_concentration
             COMMAND vtsimnx_solver_cpp_test_transport_humidity_concentration)

    # 8) acmodel: factory + logger basic behavior
    add_executable(vtsimnx_solver_cpp_test_acmodel_core
        tests_cpp/test_acmodel_core.cpp
        ../acmodel/acmodel.cpp
        ../acmodel/aircon_constants.cpp
        ../acmodel/aircon_spec.cpp
        ../acmodel/refrigerant_calculator.cpp
        ../acmodel/criepi_model.cpp
        ../acmodel/rac_model.cpp
        ../acmodel/duct_central_model.cpp
        ../acmodel/latent_evaluate_model.cpp
        ${ARCHENV_SOURCES}
    )
    target_link_libraries(vtsimnx_solver_cpp_test_acmodel_core
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_acmodel_core PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_acmodel_core COMMAND vtsimnx_solver_cpp_test_acmodel_core)

    # 9) aircon controller: ordering + on/off gating (mock model)
    add_executable(vtsimnx_solver_cpp_test_aircon_controller
        tests_cpp/test_aircon_controller.cpp
        aircon/aircon_controller.cpp
        network/thermal_network.cpp
        network/thermal_network_outputs.cpp
        network/humidity_network.cpp
        network/humidity_network_outputs.cpp
        core/thermal/thermal_solver.cpp
        core/thermal/thermal_solver_linear_direct.cpp
        core/thermal/thermal_direct_internal.cpp
        core/thermal/thermal_direct_topology.cpp
        core/thermal/thermal_direct_rhs.cpp
        core/thermal/thermal_direct_build.cpp
        core/thermal/thermal_direct_solve.cpp
        core/thermal/thermal_direct_postprocess.cpp
        ../acmodel/acmodel.cpp
        ../acmodel/aircon_constants.cpp
        ../acmodel/aircon_spec.cpp
        ../acmodel/refrigerant_calculator.cpp
        ../acmodel/criepi_model.cpp
        ../acmodel/rac_model.cpp
        ../acmodel/duct_central_model.cpp
        ../acmodel/latent_evaluate_model.cpp
        ${ARCHENV_SOURCES}
    )
    target_link_libraries(vtsimnx_solver_cpp_test_aircon_controller
        Boost::boost
    )
    target_compile_options(vtsimnx_solver_cpp_test_aircon_controller PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_aircon_controller COMMAND vtsimnx_solver_cpp_test_aircon_controller)

    # 10) app: minimal smoke test (runVtsimnxSolverApp end-to-end)
    add_executable(vtsimnx_solver_cpp_test_app_smoke
        tests_cpp/test_app_smoke.cpp
        app/vtsimnx_app.cpp
        simulation_runner.cpp
        simulation_runner_helpers.cpp
        output/artifact_io.cpp
        transport/concentration_solver.cpp
        ${PARSER_SOURCES}
        ${NETWORK_SOURCES}
        ${CORE_SOURCES}
        ${AIRCON_SOURCES}
        ${ACMODEL_SOURCES}
        ${ARCHENV_SOURCES}
    )
    target_link_libraries(vtsimnx_solver_cpp_test_app_smoke
        Boost::boost
        ceres
    )
    target_compile_options(vtsimnx_solver_cpp_test_app_smoke PRIVATE
        -Wall
        -Wextra
    )
    add_test(NAME vtsimnx_solver_cpp_test_app_smoke COMMAND vtsimnx_solver_cpp_test_app_smoke)

endif()

