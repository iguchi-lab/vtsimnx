#pragma once

#include <fstream>
#include <string>

// ログ書き込みヘルパー関数
inline void writeLog(std::ostream& logFile, const std::string& message) {
    logFile << message << std::endl;
}

