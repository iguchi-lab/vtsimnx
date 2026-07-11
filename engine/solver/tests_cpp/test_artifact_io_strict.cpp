#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "output/artifact_io.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

} // namespace

int main() {
    // schema/result サイズ不一致はエラー
    {
        std::ofstream ofs("/tmp/vtsimnx_artifact_strict.bin", std::ios::binary | std::ios::trunc);
        std::string err;
        std::vector<float> v = {1.0f, 2.0f};
        expectTrue(!ArtifactIO::writeFloat32ArrayBinary(ofs, v, 3, err),
                   "short result vs schema should fail");
        expectTrue(!err.empty(), "short result should set err");
    }
    {
        std::ofstream ofs("/tmp/vtsimnx_artifact_strict.bin", std::ios::binary | std::ios::trunc);
        std::string err;
        std::vector<float> v = {1.0f, 2.0f, 3.0f};
        expectTrue(!ArtifactIO::writeFloat32ArrayBinary(ofs, v, 2, err),
                   "long result vs schema should fail");
    }
    {
        std::ofstream ofs("/tmp/vtsimnx_artifact_strict.bin", std::ios::binary | std::ios::trunc);
        std::string err;
        std::vector<float> v = {1.0f, 2.0f};
        expectTrue(ArtifactIO::writeFloat32ArrayBinary(ofs, v, 2, err),
                   "exact size should succeed");
        expectTrue(err.empty(), "exact size should leave err empty");
    }
    {
        std::ofstream ofs("/tmp/vtsimnx_artifact_strict.bin", std::ios::binary | std::ios::trunc);
        std::string err;
        std::vector<float> empty;
        expectTrue(ArtifactIO::writeFloat32ArrayBinary(ofs, empty, 0, err),
                   "empty schema/result should succeed");
        expectTrue(!ArtifactIO::writeFloat32ArrayBinary(ofs, std::vector<float>{1.0f}, 0, err),
                   "non-empty with empty schema should fail");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}
