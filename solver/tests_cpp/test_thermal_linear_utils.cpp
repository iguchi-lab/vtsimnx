#include <iostream>
#include <string>
#include <vector>

#include "core/thermal/thermal_linear_utils.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

void expectEqInt(int actual, int expected, const std::string& msg) {
    if (actual != expected) {
        fail(msg + " (actual=" + std::to_string(actual) + ", expected=" + std::to_string(expected) + ")");
    }
}

} // namespace

int main() {
    // -----------------------------
    // RowIndexMap: build/get
    // -----------------------------
    {
        thermal_linear_utils::RowIndexMap map;
        const std::vector<int> cols = {0, 2, 5, 9, 11, 15};
        map.buildFromCols(cols);
        for (int i = 0; i < static_cast<int>(cols.size()); ++i) {
            expectEqInt(map.get(cols[static_cast<size_t>(i)]), i, "RowIndexMap: get(col) returns local index");
        }
        expectEqInt(map.get(-1), -1, "RowIndexMap: missing(-1) -> -1");
        expectEqInt(map.get(999), -1, "RowIndexMap: missing(999) -> -1");
    }

    // -----------------------------
    // RowIndexMap: larger pattern should still work
    // -----------------------------
    {
        thermal_linear_utils::RowIndexMap map;
        std::vector<int> cols;
        cols.reserve(128);
        for (int i = 0; i < 128; ++i) cols.push_back(i * 3 + 1);
        map.buildFromCols(cols);
        expectEqInt(map.get(cols.front()), 0, "RowIndexMap: large/front");
        expectEqInt(map.get(cols.back()), static_cast<int>(cols.size()) - 1, "RowIndexMap: large/back");
        expectEqInt(map.get(2), -1, "RowIndexMap: large/missing");
    }

    // -----------------------------
    // isSymmetricPatternByCols
    // -----------------------------
    {
        // symmetric (n=2)
        std::vector<std::vector<int>> colsSym = {
            {0, 1},
            {0, 1},
        };
        expectTrue(thermal_linear_utils::isSymmetricPatternByCols(colsSym),
                   "isSymmetricPatternByCols: symmetric pattern returns true");

        // non-symmetric
        std::vector<std::vector<int>> colsNon = {
            {0, 1},
            {1},
        };
        expectTrue(!thermal_linear_utils::isSymmetricPatternByCols(colsNon),
                   "isSymmetricPatternByCols: non-symmetric pattern returns false");
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


