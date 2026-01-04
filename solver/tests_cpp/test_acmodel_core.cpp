#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acmodel/acmodel.h"

namespace {

int g_failures = 0;

void fail(const std::string& msg) {
    ++g_failures;
    std::cerr << "[FAIL] " << msg << "\n";
}

void expectTrue(bool cond, const std::string& msg) {
    if (!cond) fail(msg);
}

template <class Fn>
void expectThrows(Fn fn, const std::string& msg) {
    try {
        fn();
        fail(msg + " (expected throw)");
    } catch (const std::exception&) {
        // ok
    }
}

} // namespace

int main() {
    using namespace acmodel;

    // -----------------------------
    // getTypeFromString: mapping + unknown throws
    // -----------------------------
    {
        expectTrue(AirconModelFactory::getTypeFromString("CRIEPI") == AirconType::CRIEPI, "type: CRIEPI");
        expectTrue(AirconModelFactory::getTypeFromString("RAC") == AirconType::RAC, "type: RAC");
        expectTrue(AirconModelFactory::getTypeFromString("DUCT_CENTRAL") == AirconType::DUCT_CENTRAL, "type: DUCT_CENTRAL");
        expectTrue(AirconModelFactory::getTypeFromString("LATENT_EVALUATE") == AirconType::LATENT_EVALUATE, "type: LATENT_EVALUATE");
        expectThrows([&]() { (void)AirconModelFactory::getTypeFromString("UNKNOWN"); }, "type: unknown throws");
    }

    // -----------------------------
    // logger + verbosity gating + prefix
    // -----------------------------
    {
        std::vector<std::string> captured;
        setLogger([&](const std::string& s) { captured.push_back(s); });

        setLogVerbosity(1);
        log("hello", 1);
        log("hidden", 2);
        expectTrue(captured.size() == 1, "verbosity=1 logs only level<=1");
        if (!captured.empty()) {
            expectTrue(captured[0] == std::string("　　[acmodel] ") + "hello", "logger: prefix added exactly once");
        }

        setLogVerbosity(2);
        log("show", 2);
        expectTrue(captured.size() == 2, "verbosity=2 logs level<=2");
        if (captured.size() >= 2) {
            expectTrue(captured[1] == std::string("　　[acmodel] ") + "show", "logger: prefix (level=2)");
        }

        // cleanup
        setLogger(nullptr);
    }

    if (g_failures == 0) {
        std::cout << "[OK] all tests passed\n";
        return 0;
    }
    std::cerr << "[NG] failures=" << g_failures << "\n";
    return 1;
}


