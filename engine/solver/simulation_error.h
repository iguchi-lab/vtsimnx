#pragma once

#include <stdexcept>
#include <string>

enum class SimulationErrorCode {
    PressureNotConverged,
    CouplingMaxIterations,
    ThermalNotConverged,
    AirconMaxIterations,
};

class SimulationError : public std::runtime_error {
public:
    SimulationError(SimulationErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    SimulationErrorCode code() const noexcept { return code_; }

private:
    SimulationErrorCode code_;
};
