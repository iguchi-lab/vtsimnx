#include <iostream>

#include "app/vtsimnx_app.h"

int main(int argc, char* argv[])
{
    // 引数チェック
    // 使い方: vtsimnx_solver <input.json> <output.json>
    if (argc < 3) {
        std::cerr << "使い方: " << argv[0] << " <input.json> <output.json>\n";
        return 1;
    }

    const char* inputPath  = argv[1];
    const char* outputPath = argv[2];
    return runVtsimnxSolverApp(inputPath, outputPath);
}
