#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace aircon::operating_point {

// One cooling request serves starting, holding and airflow selection. A wet
// coil's latent requirement is not an independent, zero-sensible dehumidifier.
struct CoolingRequest {
    bool temperature = false;
    bool humidity = false;
    bool reheat = false;
    bool numericalTrial = false;
    bool mayCool = true;
    bool blocked = false;
    double sensibleW = 0.0;
    double latentW = 0.0;

    bool on() const {
        return !blocked && (reheat || (mayCool &&
            (temperature || humidity || numericalTrial)));
    }
    double coolingW() const { return std::max(0.0, sensibleW) + std::max(0.0, latentW); }
};

struct Score {
    double temperatureExcess = 0.0;
    double humidityExcess = 0.0;
    bool feasible() const { return temperatureExcess <= 0.1 && humidityExcess <= 0.1; }
};

// The requested cooling temperature is an upper constraint. Within that
// envelope, reduce RH shortfall. Equivalent points retain the first solution;
// electricity is an output, not a control objective. Never reward RH that
// fell only because an underpowered candidate let the room overheat.
inline bool better(const Score& a, const Score& b) {
    if (a.feasible() != b.feasible()) return a.feasible();
    const double at = std::max(0.0, a.temperatureExcess - 0.1);
    const double bt = std::max(0.0, b.temperatureExcess - 0.1);
    if (std::abs(at - bt) > 0.01) return at < bt;
    const double ah = std::max(0.0, a.humidityExcess - 0.1);
    const double bh = std::max(0.0, b.humidityExcess - 0.1);
    if (std::abs(ah - bh) > 0.01) return ah < bh;
    return false;
}

struct AirflowSearch {
    bool seededFromPrevious = false;
    bool initialized = false;
    bool complete = false;
    bool restoring = false;
    bool expanded = false;
    double ratio = 1.0;
    double minimumRatio = 0.1;
    double bestRatio = 1.0;
    bool bestOn = true;
    bool bestHumidityHold = false;
    bool bestCapacityLimited = false;
    bool bestFanTrialDone = false;
    double bestSetpoint = 0.0;
    Score best{std::numeric_limits<double>::infinity(),
               std::numeric_limits<double>::infinity()};
    std::vector<double> tried;
    std::vector<double> pending;

    void enqueue(double value) {
        const double r = std::clamp(value, minimumRatio, 1.0);
        for (double old : tried) if (std::abs(r-old) < 1e-3) return;
        for (double old : pending) if (std::abs(r-old) < 1e-3) return;
        pending.push_back(r);
    }
};

} // namespace aircon::operating_point
