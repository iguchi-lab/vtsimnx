#pragma once

#include <string>
#include <vector>
#include <chrono>

struct TimingEntry {
    std::string name;
    double durationMs = 0.0;
    std::string metadata;
};

using TimingList = std::vector<TimingEntry>;

// タイミング計測の有効/無効（デフォルト: 有効）
// 実運用では「詳細タイミングを無効化」してオーバーヘッドを抑えたいケースがあるため、
// グローバルフラグで制御できるようにしておく（C++17 inline 変数）。
inline bool g_enable_timings = true;
inline void setTimingsEnabled(bool enabled) { g_enable_timings = enabled; }

class ScopedTimer {
public:
    ScopedTimer(TimingList& timings, std::string name, std::string metadata = {})
        : timings_(g_enable_timings ? &timings : nullptr) {
        if (!timings_) return;
        // 有効時のみ計測・保持する（無効時は now() すら呼ばない）
        name_ = std::move(name);
        metadata_ = std::move(metadata);
        start_ = std::chrono::steady_clock::now();
    }

    ~ScopedTimer() {
        if (!timings_) return;
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start_).count();
        timings_->push_back(TimingEntry{std::move(name_), ms, std::move(metadata_)});
    }

private:
    TimingList* timings_ = nullptr;
    std::string name_;
    std::string metadata_;
    std::chrono::steady_clock::time_point start_{};
};


