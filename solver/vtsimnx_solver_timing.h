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

class ScopedTimer {
public:
    ScopedTimer(TimingList& timings, std::string name, std::string metadata = {})
        : timings_(timings),
          name_(std::move(name)),
          metadata_(std::move(metadata)),
          start_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start_).count();
        timings_.push_back(TimingEntry{name_, ms, metadata_});
    }

private:
    TimingList& timings_;
    std::string name_;
    std::string metadata_;
    std::chrono::steady_clock::time_point start_;
};


