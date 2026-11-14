cmake_minimum_required(VERSION 3.13)
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

set(VENTILATION_PRESSURE_SOURCES
    core/ventilation/pressure_solver.cpp
    core/ventilation/pressure_solver_ceres.cpp
    core/ventilation/pressure_solver_fallback.cpp
    core/ventilation/pressure_solver_fallback_partition.cpp
    core/ventilation/pressure_solver_fallback_stage_a.cpp
    core/ventilation/pressure_solver_fallback_mutation.cpp
    core/ventilation/pressure_solver_fallback_stage_b.cpp
    core/ventilation/pressure_solver_fallback_decision.cpp
    core/ventilation/pressure_constraints.cpp
)

set(CORE_SOURCES
    ${VENTILATION_PRESSURE_SOURCES}
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
    aircon/aircon_power.cpp
    aircon/aircon_onoff.cpp
    aircon/aircon_latent.cpp
    aircon/aircon_capacity.cpp
    aircon/aircon_airflow.cpp
    aircon/aircon_network_utils.cpp
    hrv/hrv_controller.cpp
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
    simulation_runner.cpp
    simulation_runner_helpers.cpp
    simulation_coupled_step.cpp
    simulation_coupling_control.cpp
    simulation_timestep_result.cpp
    simulation_aircon_iteration.cpp
    simulation_inner_coupling.cpp
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

# Compile numerical code once, and share it with the CLI and every C++ test.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/../build)
find_package(Boost REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
option(VTSIMNX_ENABLE_KLU "Enable KLU backend for thermal direct solver" ON)
option(VTSIMNX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(VTSIMNX_BUILD_CPP_TESTS "Build C++ unit tests for solver" ON)

add_library(vtsimnx_core STATIC ${ALL_SOURCES})
target_link_libraries(vtsimnx_core PUBLIC Boost::boost ceres Eigen3::Eigen)
target_compile_options(vtsimnx_core PRIVATE -Wall -Wextra)
if(VTSIMNX_ENABLE_KLU)
    find_path(KLU_INCLUDE_DIR NAMES klu.h PATH_SUFFIXES suitesparse)
    find_library(KLU_LIBRARY NAMES klu)
    if(KLU_INCLUDE_DIR AND KLU_LIBRARY)
        target_include_directories(vtsimnx_core PUBLIC ${KLU_INCLUDE_DIR})
        target_compile_definitions(vtsimnx_core PUBLIC VTSIMNX_USE_KLU=1)
        target_link_libraries(vtsimnx_core PUBLIC ${KLU_LIBRARY})
        foreach(dependency suitesparseconfig amd colamd btf)
            find_library(${dependency}_LIBRARY NAMES ${dependency})
            if(${dependency}_LIBRARY)
                target_link_libraries(vtsimnx_core PUBLIC ${${dependency}_LIBRARY})
            endif()
        endforeach()
    else()
        message(STATUS "KLU not found; using Eigen::SparseLU")
    endif()
endif()
if(VTSIMNX_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "Sanitizers require GCC or Clang")
    endif()
    target_compile_options(vtsimnx_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(vtsimnx_core PUBLIC -fsanitize=address,undefined)
endif()
add_executable(vtsimnx_solver vtsimnx_solver.cpp)
target_link_libraries(vtsimnx_solver PRIVATE vtsimnx_core)

if(VTSIMNX_BUILD_CPP_TESTS)
    enable_testing()
    set(SOLVER_CPP_TESTS
        flow_math
        refrigerant_calculator
        fan_jacobian_sweep
        parser_thermal_branches
        parser_branches_validation
        parser_nodes_mode
        parser_nodes_required
        thermal_linear_utils
        thermal_direct_cache
        thermal_advection_sign
        moist_enthalpy_advection
        thermal_enable_response
        pressure_constraints
        pressure_solver_safety
        pressure_acceptance_coupling
        vent_parallel_branch_flow_rates
        flow_rate_key_collision
        network_rebuild
        transport_humidity_concentration
        humidity_enable_zero_cap
        core_physics_known_solutions
        acmodel_core
        aircon_controller
        hrv_controller
        app_smoke
        artifact_io_strict
        simulation_control_flow
        simulation_runner_integration
    )
    foreach(name IN LISTS SOLVER_CPP_TESTS)
        set(target vtsimnx_solver_cpp_test_${name})
        add_executable(${target} tests_cpp/test_${name}.cpp)
        target_link_libraries(${target} PRIVATE vtsimnx_core)
        target_compile_options(${target} PRIVATE -Wall -Wextra)
        add_test(NAME ${target} COMMAND ${target})
    endforeach()
    target_compile_definitions(vtsimnx_solver_cpp_test_acmodel_core PRIVATE
        RAC_REFERENCE_FULL_TSV_PATH="${CMAKE_CURRENT_SOURCE_DIR}/tests_cpp/data/rac_reference_full.tsv")
endif()
