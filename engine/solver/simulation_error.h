#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace simulation {

enum class ErrorCode {
    PressureNotConverged,
    CouplingMaxIterations,
    ThermalNotConverged,
    HumidityNotConverged,
    AirconMaxIterations,
};

// API / manifest 向けの安定な snake_case 文字列。
inline std::string_view toErrorCodeString(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::PressureNotConverged:
        return "pressure_not_converged";
    case ErrorCode::CouplingMaxIterations:
        return "coupling_max_iterations";
    case ErrorCode::ThermalNotConverged:
        return "thermal_not_converged";
    case ErrorCode::HumidityNotConverged:
        return "humidity_not_converged";
    case ErrorCode::AirconMaxIterations:
        return "aircon_max_iterations";
    }
    return "solver_error";
}

class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message)
        : std::runtime_error(message), code_(code) {}

    ErrorCode code() const noexcept { return code_; }

private:
    ErrorCode code_;
};

} // namespace simulation
