#pragma once

#include <algorithm>
#include <fstream>
#include <string>
#include <string_view>
#include <ios>

namespace logging_detail {
// std::ostream* をキーにした map は高頻度ログで重くなりやすい。
// iword/xalloc を使って ostream 本体に状態を保持する（ヒープ確保・ハッシュを避ける）。
inline int indentIndex() {
    static int idx = std::ios_base::xalloc();
    return idx;
}
inline int timestepIndex() {
    static int idx = std::ios_base::xalloc();
    return idx;
}
inline long& indentLevel(std::ios_base& ios) {
    return ios.iword(indentIndex()); // default 0
}
inline long& timestepMeta(std::ios_base& ios) {
    return ios.iword(timestepIndex()); // default 0 (未設定は -1 を使う)
}
} // namespace logging_detail

inline void setLogTimestepMeta(std::ostream& os, int timestepIndex) {
    logging_detail::timestepMeta(os) = static_cast<long>(timestepIndex);
}

inline void clearLogTimestepMeta(std::ostream& os) {
    logging_detail::timestepMeta(os) = -1;
}

inline void writeLog(std::ostream& logFile,
                     const std::string& message,
                     bool includeTimestepMeta);

class ScopedLogIndent {
public:
    ScopedLogIndent(std::ostream& os, int depth = 1)
        : os_(&os), depth_(std::max(0, depth)) {
        logging_detail::indentLevel(*os_) += depth_;
    }

    ScopedLogIndent(const ScopedLogIndent&) = delete;
    ScopedLogIndent& operator=(const ScopedLogIndent&) = delete;

    ScopedLogIndent(ScopedLogIndent&& other) noexcept
        : os_(other.os_), depth_(other.depth_) {
        other.os_ = nullptr;
        other.depth_ = 0;
    }

    ScopedLogIndent& operator=(ScopedLogIndent&& other) noexcept {
        if (this != &other) {
            release();
            os_ = other.os_;
            depth_ = other.depth_;
            other.os_ = nullptr;
            other.depth_ = 0;
        }
        return *this;
    }

    ~ScopedLogIndent() {
        release();
    }

private:
    void release() {
        if (!os_ || depth_ == 0) return;
        auto& indent = logging_detail::indentLevel(*os_);
        indent = std::max(0L, indent - depth_);
        depth_ = 0;
        os_ = nullptr;
    }

    std::ostream* os_;
    int depth_;
};

class ScopedLogSection {
public:
    ScopedLogSection(std::ostream& os,
                     const std::string& title,
                     bool includeTimestepMeta = false)
        : os_(&os), active_(true) {
        writeLog(*os_, title, includeTimestepMeta);
        logging_detail::indentLevel(*os_) += 1;
    }

    ScopedLogSection(const ScopedLogSection&) = delete;
    ScopedLogSection& operator=(const ScopedLogSection&) = delete;

    ScopedLogSection(ScopedLogSection&& other) noexcept
        : os_(other.os_), active_(other.active_) {
        other.os_ = nullptr;
        other.active_ = false;
    }

    ScopedLogSection& operator=(ScopedLogSection&& other) noexcept {
        if (this != &other) {
            release();
            os_ = other.os_;
            active_ = other.active_;
            other.os_ = nullptr;
            other.active_ = false;
        }
        return *this;
    }

    ~ScopedLogSection() {
        release();
    }

private:
    void release() {
        if (!active_ || !os_) return;
        auto& indent = logging_detail::indentLevel(*os_);
        indent = std::max(0L, indent - 1);
        active_ = false;
        os_ = nullptr;
    }

    std::ostream* os_;
    bool active_;
};

inline void writeLog(std::ostream& logFile,
                     const std::string& message,
                     bool includeTimestepMeta = false) {
    std::string_view content(message);

    // レガシーなハイフンによるインデントを除去（2本以上のハイフンのみ）
    size_t hyphenCount = 0;
    while (hyphenCount < content.size() && content[hyphenCount] == '-') {
        hyphenCount++;
    }
    if (hyphenCount >= 2) {
        size_t pos = hyphenCount;
        while (pos < content.size() && content[pos] == ' ') {
            pos++;
        }
        content.remove_prefix(pos);
    }

    // NOTE: std::endl は毎回 flush するため、ログ量が多いと支配的なボトルネックになる。
    // ここでは '\n' のみにして flush は呼び出し側（必要時）に任せる。
    const long ts = logging_detail::timestepMeta(logFile);
    if (includeTimestepMeta && ts >= 0) {
        logFile << "[ts=" << ts << "] ";
    }
    const long indent = logging_detail::indentLevel(logFile);
    for (long i = 0; i < indent; ++i) {
        logFile << "  ";
    }
    logFile << content << '\n';
}

