#pragma once

#include <algorithm>
#include <fstream>
#include <string>
#include <string_view>
#include <unordered_map>

namespace logging_detail {
struct LogState {
    int indentLevel = 0;
    int timestepMeta = -1;
};

inline LogState& stateFor(std::ostream& os) {
    static std::unordered_map<std::ostream*, LogState> states;
    return states[&os];
}
} // namespace logging_detail

inline void setLogTimestepMeta(std::ostream& os, int timestepIndex) {
    logging_detail::stateFor(os).timestepMeta = timestepIndex;
}

inline void clearLogTimestepMeta(std::ostream& os) {
    logging_detail::stateFor(os).timestepMeta = -1;
}

inline void writeLog(std::ostream& logFile,
                     const std::string& message,
                     bool includeTimestepMeta);

class ScopedLogIndent {
public:
    ScopedLogIndent(std::ostream& os, int depth = 1)
        : os_(&os), depth_(std::max(0, depth)) {
        logging_detail::stateFor(*os_).indentLevel += depth_;
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
        auto& state = logging_detail::stateFor(*os_);
        state.indentLevel = std::max(0, state.indentLevel - depth_);
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
        logging_detail::stateFor(*os_).indentLevel++;
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
        auto& state = logging_detail::stateFor(*os_);
        state.indentLevel = std::max(0, state.indentLevel - 1);
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

    auto& state = logging_detail::stateFor(logFile);
    if (includeTimestepMeta && state.timestepMeta >= 0) {
        logFile << "[ts=" << state.timestepMeta << "] ";
    }
    for (int i = 0; i < state.indentLevel; ++i) {
        logFile << "  ";
    }
    logFile << content << std::endl;
}

