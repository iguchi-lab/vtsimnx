#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// 熱ソルバ（線形）の内部ユーティリティ。
// ※どの翻訳単位からインクルードされても良いように、全て inline で定義する。

namespace thermal_linear_utils {

// rowColsPattern と同型の高速 lookup（col -> local index）
// 行の列数は小さい（典型 10〜30）ため、小さなオープンアドレス法で O(1) 参照する。
struct RowIndexMap {
    int mask = 0;                 // tableSize-1（2冪）
    std::vector<int> keys;        // col (param idx) / empty=-1
    std::vector<int> values;      // local index in row (0..rowLen-1) / empty=-1

    void clear() {
        mask = 0;
        keys.clear();
        values.clear();
    }

    static int nextPow2(int x) {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }

    void buildFromCols(const std::vector<int>& cols) {
        const int tableSize = nextPow2(std::max(2, static_cast<int>(cols.size()) * 2));
        mask = tableSize - 1;
        keys.assign(static_cast<size_t>(tableSize), -1);
        values.assign(static_cast<size_t>(tableSize), -1);
        for (int local = 0; local < static_cast<int>(cols.size()); ++local) {
            const int col = cols[static_cast<size_t>(local)];
            std::uint32_t h = static_cast<std::uint32_t>(col) * 2654435761u;
            int idx = static_cast<int>(h) & mask;
            // 空きスロットを見つける（cols は unique 想定）
            while (keys[static_cast<size_t>(idx)] != -1) idx = (idx + 1) & mask;
            keys[static_cast<size_t>(idx)] = col;
            values[static_cast<size_t>(idx)] = local;
        }
    }

    inline int get(int col) const {
        const int tableSize = mask + 1;
        if (tableSize <= 0) return -1;
        std::uint32_t h = static_cast<std::uint32_t>(col) * 2654435761u;
        int idx = static_cast<int>(h) & mask;
        while (keys[static_cast<size_t>(idx)] != -1) {
            if (keys[static_cast<size_t>(idx)] == col) return values[static_cast<size_t>(idx)];
            idx = (idx + 1) & mask;
        }
        return -1;
    }
};

inline std::uint64_t fnv1a64_update(std::uint64_t h, std::uint64_t v) {
    constexpr std::uint64_t kFnvOffset = 14695981039346656037ull;
    constexpr std::uint64_t kFnvPrime  = 1099511628211ull;
    if (h == 0) h = kFnvOffset;
    h ^= v;
    h *= kFnvPrime;
    return h;
}

inline std::uint64_t hashDoubleBits(std::uint64_t h, double x) {
    std::uint64_t bits = 0;
    static_assert(sizeof(double) == sizeof(std::uint64_t), "double size mismatch");
    std::memcpy(&bits, &x, sizeof(bits));
    return fnv1a64_update(h, bits);
}

// 非ゼロパターンが対称か（A(i,j) があれば A(j,i) もあるか）だけを見る。
inline bool isSymmetricPatternByCols(const std::vector<std::vector<int>>& colIndices) {
    const int n = static_cast<int>(colIndices.size());
    for (int r = 0; r < n; ++r) {
        const auto& colsR = colIndices[static_cast<size_t>(r)];
        for (int c : colsR) {
            if (c == r) continue;
            if (c < 0 || c >= n) return false;
            const auto& colsC = colIndices[static_cast<size_t>(c)];
            auto it = std::lower_bound(colsC.begin(), colsC.end(), r);
            if (it == colsC.end() || *it != r) return false;
        }
    }
    return true;
}

} // namespace thermal_linear_utils


