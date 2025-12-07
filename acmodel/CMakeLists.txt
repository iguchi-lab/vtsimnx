# acmodel モジュールのCMakeList

# archenvライブラリの追加
add_subdirectory(${CMAKE_SOURCE_DIR}/archenv archenv_build)

# ヘッダーファイル
set(ACMODEL_HEADERS
    include/aircon_constants.h
    include/aircon_spec.h
    include/refrigerant_calculator.h
    include/criepi_model.h
    include/rac_model.h
    include/duct_central_model.h
    include/latent_evaluate_model.h
    include/acmodel.h
)

# ソースファイル
set(ACMODEL_SOURCES
    src/aircon_constants.cpp
    src/aircon_spec.cpp
    src/refrigerant_calculator.cpp
    src/criepi_model.cpp
    src/rac_model.cpp
    src/duct_central_model.cpp
    src/latent_evaluate_model.cpp
    src/acmodel.cpp
)

# acmodel静的ライブラリを作成
add_library(acmodel STATIC ${ACMODEL_SOURCES} ${ACMODEL_HEADERS})

# インクルードディレクトリを設定
target_include_directories(acmodel PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/archenv/include  # archenvヘッダーを追加
)

# ライブラリのリンク
find_package(nlohmann_json REQUIRED)
target_link_libraries(acmodel PRIVATE 
    nlohmann_json::nlohmann_json
    archenv  # archenvライブラリをリンク
)

# コンパイラオプション
target_compile_features(acmodel PRIVATE cxx_std_17)

# ヘッダーファイルをプロジェクトに含める（IDEでの表示用）
source_group("Header Files" FILES ${ACMODEL_HEADERS})
source_group("Source Files" FILES ${ACMODEL_SOURCES}) 