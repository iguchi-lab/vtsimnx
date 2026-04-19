#include "core/thermal/thermal_direct_internal.h"

#include <cstdlib>

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
#include <klu.h>
#endif

namespace ThermalSolverLinearDirect::detail {

namespace {

constexpr const char* kLuBackendEnv = "VTSIMNX_THERMAL_DIRECT_LU";

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
struct KluCache {
    bool analyzed = false;
    bool factorized = false;
    int n = 0;
    size_t nnz = 0;
    std::uint64_t patternHash = 0;
    std::uint64_t valueHash = 0;
    klu_symbolic* symbolic = nullptr;
    klu_numeric* numeric = nullptr;
    klu_common common{};
};

KluCache g_kluCache;

void clearKluNumeric() {
    if (g_kluCache.numeric != nullptr) {
        klu_free_numeric(&g_kluCache.numeric, &g_kluCache.common);
        g_kluCache.numeric = nullptr;
    }
    g_kluCache.factorized = false;
}

void clearKluAll() {
    clearKluNumeric();
    if (g_kluCache.symbolic != nullptr) {
        klu_free_symbolic(&g_kluCache.symbolic, &g_kluCache.common);
        g_kluCache.symbolic = nullptr;
    }
    g_kluCache.analyzed = false;
    g_kluCache.n = 0;
    g_kluCache.nnz = 0;
    g_kluCache.patternHash = 0;
    g_kluCache.valueHash = 0;
    klu_defaults(&g_kluCache.common);
}
#endif

bool shouldUseKluBackend(std::ostream& logFile) {
    static int backendState = -1; // -1: unresolved, 0: Eigen, 1: KLU
    if (backendState >= 0) return backendState == 1;

    const char* env = std::getenv(kLuBackendEnv);
    const std::string requested = (env != nullptr) ? std::string(env) : std::string();
    const bool forceLu = (requested == "lu" || requested == "LU" ||
                          requested == "eigen" || requested == "EIGEN" ||
                          requested == "sparselu" || requested == "SPARSELU");
    const bool forceKlu = (requested == "klu" || requested == "KLU");
    if (forceLu) {
        backendState = 0;
        writeLog(logFile, "--------疎直接法(DirectT): LU backend=Eigen::SparseLU (forced by env)");
        return false;
    }

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    if (!requested.empty() && !forceKlu) {
        backendState = 1;
        writeLog(logFile, "--------疎直接法(DirectT): unknown LU backend env; use default KLU");
        return true;
    }
    backendState = 1;
    writeLog(logFile, "--------疎直接法(DirectT): LU backend=KLU");
    return true;
#else
    backendState = 0;
    if (forceKlu) {
        writeLog(logFile, "--------疎直接法(DirectT): KLU requested but unavailable, fallback to Eigen::SparseLU");
    } else {
        writeLog(logFile, "--------疎直接法(DirectT): LU backend=Eigen::SparseLU (KLU not available)");
    }
    return false;
#endif
}

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
bool ensureKluPattern(const Eigen::SparseMatrix<double>& A, int n, size_t nnz, std::uint64_t patternHash, std::ostream& logFile) {
    const bool needAnalyze = (!g_kluCache.analyzed) ||
                             (g_kluCache.n != n) ||
                             (g_kluCache.nnz != nnz) ||
                             (g_kluCache.patternHash != patternHash);
    if (!needAnalyze) return true;

    clearKluAll();
    g_kluCache.n = n;
    g_kluCache.nnz = nnz;
    g_kluCache.patternHash = patternHash;
    g_kluCache.symbolic = klu_analyze(n, const_cast<int*>(A.outerIndexPtr()), const_cast<int*>(A.innerIndexPtr()), &g_kluCache.common);
    if (g_kluCache.symbolic == nullptr) {
        writeLog(logFile, "--------疎直接法(DirectT): KLU analyze failed");
        return false;
    }
    g_kluCache.analyzed = true;
    return true;
}

bool factorizeWithKlu(const Eigen::SparseMatrix<double>& A, std::uint64_t valueHash, std::ostream& logFile) {
    if (!g_kluCache.analyzed) return false;
    if (g_kluCache.factorized && g_kluCache.valueHash == valueHash) return true;

    clearKluNumeric();
    g_kluCache.numeric = klu_factor(const_cast<int*>(A.outerIndexPtr()),
                                    const_cast<int*>(A.innerIndexPtr()),
                                    const_cast<double*>(A.valuePtr()),
                                    g_kluCache.symbolic,
                                    &g_kluCache.common);
    if (g_kluCache.numeric == nullptr) {
        writeLog(logFile, "--------疎直接法(DirectT): KLU factorize failed (singular/ill-conditioned)");
        return false;
    }
    g_kluCache.factorized = true;
    g_kluCache.valueHash = valueHash;
    return true;
}

bool solveWithKlu(const Eigen::VectorXd& b, Eigen::VectorXd& sol, std::ostream& logFile) {
    if (!g_kluCache.analyzed || !g_kluCache.factorized) return false;
    sol = b;
    const int ok = klu_solve(g_kluCache.symbolic, g_kluCache.numeric, g_kluCache.n, 1, sol.data(), &g_kluCache.common);
    if (ok == 0) {
        writeLog(logFile, "--------疎直接法(DirectT): KLU solve failed");
        return false;
    }
    return true;
}
#endif

} // namespace

bool solveSparseDirect(const LinearSystem& system,
                       std::vector<double>& x,
                       double tolerance,
                       std::ostream& logFile,
                       std::string& methodLabel) {
    using thermal_linear_utils::fnv1a64_update;
    using thermal_linear_utils::hashDoubleBits;
    using thermal_linear_utils::isSymmetricPatternByCols;

    const size_t n = x.size();
    if (n == 0) return true;

    size_t nnz = 0;
    for (size_t i = 0; i < n; ++i) nnz += system.colIndices[i].size();

    std::uint64_t patternHash = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto& cols = system.colIndices[i];
        for (size_t k = 0; k < cols.size(); ++k) {
            patternHash = fnv1a64_update(patternHash,
                                         (static_cast<std::uint64_t>(i) << 32) ^ static_cast<std::uint64_t>(cols[k]));
        }
    }

    Eigen::VectorXd b(static_cast<int>(n));
    for (size_t i = 0; i < n; ++i) b[static_cast<int>(i)] = system.b[i];
    bool useKluBackend = shouldUseKluBackend(logFile);

    const bool needRebuildPattern = (!g_sparseLuCache.analyzed) ||
                                   (g_sparseLuCache.n != static_cast<int>(n)) ||
                                   (g_sparseLuCache.nnz != nnz) ||
                                   (g_sparseLuCache.patternHash != patternHash);

    if (needRebuildPattern) {
        ++g_directTStats.patternRebuild;
        g_sparseLuCache.analyzed = false;
        g_sparseLuCache.n = static_cast<int>(n);
        g_sparseLuCache.nnz = nnz;
        g_sparseLuCache.patternHash = patternHash;
        g_sparseLuCache.factorized = false;
        g_sparseLuCache.valueHash = 0;
        g_sparseLuCache.valuePtrIndexByRow.clear();
        g_sparseLuCache.A.resize(0, 0);

        g_sparseLuCache.solver.~SparseLU();
        new (&g_sparseLuCache.solver) Eigen::SparseLU<Eigen::SparseMatrix<double>>();

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(nnz);
        std::uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& cols = system.colIndices[i];
            const auto& vals = system.A[i];
            for (size_t k = 0; k < cols.size(); ++k) {
                triplets.emplace_back(static_cast<int>(i), cols[k], vals[k]);
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }
        g_sparseLuCache.A = Eigen::SparseMatrix<double>(static_cast<int>(n), static_cast<int>(n));
        g_sparseLuCache.A.setFromTriplets(triplets.begin(), triplets.end());
        g_sparseLuCache.A.makeCompressed();

        // valuePtr mapping (row-wise)
        g_sparseLuCache.valuePtrIndexByRow.assign(n, {});
        std::vector<std::vector<std::pair<int, int>>> rowEntries(n);
        for (size_t r = 0; r < n; ++r) rowEntries[r].reserve(system.colIndices[r].size());
        double* base = g_sparseLuCache.A.valuePtr();
        for (int outer = 0; outer < g_sparseLuCache.A.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(g_sparseLuCache.A, outer); it; ++it) {
                const int r = it.row();
                const int c = it.col();
                const int p = static_cast<int>(&it.valueRef() - base);
                if (r >= 0 && r < static_cast<int>(n)) rowEntries[static_cast<size_t>(r)].emplace_back(c, p);
            }
        }
        for (size_t r = 0; r < n; ++r) {
            auto& entries = rowEntries[r];
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
            const auto& cols = system.colIndices[r];
            g_sparseLuCache.valuePtrIndexByRow[r].assign(cols.size(), -1);
            size_t j = 0;
            for (size_t k = 0; k < cols.size(); ++k) {
                const int col = cols[k];
                while (j < entries.size() && entries[j].first < col) ++j;
                if (j < entries.size() && entries[j].first == col) g_sparseLuCache.valuePtrIndexByRow[r][k] = entries[j].second;
            }
        }
        bool mappingOk = true;
        for (size_t r = 0; r < n && mappingOk; ++r) {
            for (int p : g_sparseLuCache.valuePtrIndexByRow[r]) {
                if (p < 0) { mappingOk = false; break; }
            }
        }
        if (!mappingOk) {
            writeLog(logFile, "--------疎直接法(DirectT): valuePtrIndexByRow の構築に失敗（パターン不一致）。停止します。");
            g_sparseLuCache.analyzed = false;
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = 0;
            g_sparseLuCache.valuePtrIndexByRow.clear();
            g_cholCache.analyzed = false;
            g_cholCache.factorized = false;
            return false;
        }

        if (useKluBackend) {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (!ensureKluPattern(g_sparseLuCache.A, static_cast<int>(n), nnz, patternHash, logFile)) {
                writeLog(logFile, "--------疎直接法(DirectT): fallback to Eigen::SparseLU after KLU analyze failure");
                useKluBackend = false;
            }
            if (useKluBackend) {
                g_sparseLuCache.analyzed = true;
            } else {
                g_sparseLuCache.solver.analyzePattern(g_sparseLuCache.A);
                g_sparseLuCache.analyzed = true;
            }
#else
            g_sparseLuCache.analyzed = false;
            return false;
#endif
        } else {
            g_sparseLuCache.solver.analyzePattern(g_sparseLuCache.A);
            g_sparseLuCache.analyzed = true;
        }
        g_sparseLuCache.valueHash = valueHash;

        g_cholCache.analyzed = false;
        g_cholCache.factorized = false;
        g_cholCache.patternSymmetric = isSymmetricPatternByCols(system.colIndices);
    } else {
        std::uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& vals = system.A[i];
            for (size_t k = 0; k < vals.size(); ++k) {
                const int p = g_sparseLuCache.valuePtrIndexByRow[i][k];
                if (p < 0) return false;
                g_sparseLuCache.A.valuePtr()[p] = vals[k];
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }
        if (g_sparseLuCache.valueHash != valueHash) {
            g_sparseLuCache.factorized = false;
            g_sparseLuCache.valueHash = valueHash;
            g_cholCache.factorized = false;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            g_kluCache.factorized = false;
#endif
        }
    }

    const bool symmetricCandidate = g_cholCache.patternSymmetric;
    Eigen::VectorXd sol;
    bool solved = false;

    if (symmetricCandidate) {
        const bool needAnalyze = (!g_cholCache.analyzed) ||
                                 (g_cholCache.n != static_cast<int>(n)) ||
                                 (g_cholCache.nnz != nnz) ||
                                 (g_cholCache.patternHash != patternHash);
        if (needAnalyze) {
            g_cholCache.analyzed = false;
            g_cholCache.n = static_cast<int>(n);
            g_cholCache.nnz = nnz;
            g_cholCache.patternHash = patternHash;
            g_cholCache.factorized = false;
            g_cholCache.valueHash = 0;
            g_cholCache.llt.analyzePattern(g_sparseLuCache.A);
            g_cholCache.ldlt.analyzePattern(g_sparseLuCache.A);
            g_cholCache.analyzed = true;
        }

        if (!g_cholCache.factorized || g_cholCache.valueHash != g_sparseLuCache.valueHash) {
            ++g_directTStats.cholFactorize;
            g_cholCache.llt.factorize(g_sparseLuCache.A);
            if (g_cholCache.llt.info() == Eigen::Success) {
                g_cholCache.factorized = true;
                g_cholCache.valueHash = g_sparseLuCache.valueHash;
                sol = g_cholCache.llt.solve(b);
                if (g_cholCache.llt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LLT";
                }
            } else {
                writeLog(logFile, "--------疎直接法(DirectT): LLT factorize failed (matrix not SPD or ill-conditioned)");
            }
            if (!solved) {
                ++g_directTStats.cholFactorize;
                g_cholCache.ldlt.factorize(g_sparseLuCache.A);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    g_cholCache.factorized = true;
                    g_cholCache.valueHash = g_sparseLuCache.valueHash;
                    sol = g_cholCache.ldlt.solve(b);
                    if (g_cholCache.ldlt.info() == Eigen::Success) {
                        solved = true;
                        methodLabel = "LDLT";
                    }
                } else {
                    writeLog(logFile, "--------疎直接法(DirectT): LDLT factorize failed (matrix not SPD / indefinite / ill-conditioned)");
                }
            }
            if (!solved) {
                g_cholCache.factorized = false;
                g_cholCache.patternSymmetric = false;
            }
        } else {
            sol = g_cholCache.llt.solve(b);
            if (g_cholCache.llt.info() == Eigen::Success) {
                solved = true;
                methodLabel = "LLT(cached)";
            } else {
                sol = g_cholCache.ldlt.solve(b);
                if (g_cholCache.ldlt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LDLT(cached)";
                }
            }
        }
    }

    if (!solved) {
        if (!g_sparseLuCache.factorized) {
            ++g_directTStats.luFactorize;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (useKluBackend) {
                if (!factorizeWithKlu(g_sparseLuCache.A, g_sparseLuCache.valueHash, logFile)) {
                    writeLog(logFile, "--------疎直接法(DirectT): fallback to Eigen::SparseLU after KLU factorize failure");
                    useKluBackend = false;
                }
            }
            if (!useKluBackend)
#endif
            {
                g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                if (g_sparseLuCache.solver.info() != Eigen::Success) {
                    writeLog(logFile, "--------疎直接法(DirectT): LU factorize failed (singular/ill-conditioned)");
                    return false;
                }
            }
            g_sparseLuCache.factorized = true;
        }
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
        if (useKluBackend) {
            if (solveWithKlu(b, sol, logFile)) {
                methodLabel = "KLU";
            } else {
                writeLog(logFile, "--------疎直接法(DirectT): fallback to Eigen::SparseLU after KLU solve failure");
                useKluBackend = false;
                g_sparseLuCache.factorized = false;
            }
        }
        if (!useKluBackend)
#endif
        {
            if (!g_sparseLuCache.factorized) {
                ++g_directTStats.luFactorize;
                g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                if (g_sparseLuCache.solver.info() != Eigen::Success) {
                    writeLog(logFile, "--------疎直接法(DirectT): LU factorize failed (fallback path)");
                    return false;
                }
                g_sparseLuCache.factorized = true;
            }
            sol = g_sparseLuCache.solver.solve(b);
            if (g_sparseLuCache.solver.info() != Eigen::Success) {
                writeLog(logFile, "--------疎直接法(DirectT): LU solve failed");
                return false;
            }
            methodLabel = "LU";
        }
    }

    // 残差チェック:
    // これまで max(|Ax-b|) を「絶対値」で判定していたが、
    // 系のスケール（b や A の係数）が大きい/小さいケースで過剰に厳しくなる。
    // tolerance は simulation.tolerance.thermal（収束判定にも使う）で、実務上は相対誤差が欲しい。
    // そこで b のスケール（max|b|）で正規化した閾値を併用する。
    Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
    const double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(maxResidual)) {
        writeLog(logFile, "--------疎直接法(DirectT): residual is not finite");
        return false;
    }
    const double bScale = (b.size() > 0) ? b.cwiseAbs().maxCoeff() : 0.0;
    const double scaledTol = std::max(1.0, bScale) * tolerance * 10.0;
    if (maxResidual > scaledTol) {
        // 重要:
        // 対称候補として LLT/LDLT が「成功」しても、数値誤差で残差が大きいことがある。
        // その場合、より頑健な LU にフォールバックすると収束するケースがある（aircon無しで顕在化しやすい）。
        const bool usedCholesky =
            (methodLabel.rfind("LLT", 0) == 0) || (methodLabel.rfind("LDLT", 0) == 0);

        auto tryLuFallback = [&](Eigen::VectorXd& ioSol, std::string& ioMethod) -> bool {
            Eigen::VectorXd sol2;
            if (!g_sparseLuCache.factorized) {
                ++g_directTStats.luFactorize;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
                if (useKluBackend) {
                    if (!factorizeWithKlu(g_sparseLuCache.A, g_sparseLuCache.valueHash, logFile)) {
                        writeLog(logFile, "--------疎直接法(DirectT): fallback to Eigen::SparseLU after KLU factorize retry failure");
                        useKluBackend = false;
                    }
                }
                if (!useKluBackend)
#endif
                {
                    g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                    if (g_sparseLuCache.solver.info() != Eigen::Success) {
                        writeLog(logFile, "--------疎直接法(DirectT): LU factorize failed on retry");
                        return false;
                    }
                }
                g_sparseLuCache.factorized = true;
            }
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (useKluBackend) {
                if (!solveWithKlu(b, sol2, logFile)) {
                    writeLog(logFile, "--------疎直接法(DirectT): fallback to Eigen::SparseLU after KLU solve retry failure");
                    useKluBackend = false;
                    g_sparseLuCache.factorized = false;
                }
            }
            if (!useKluBackend)
#endif
            {
                if (!g_sparseLuCache.factorized) {
                    ++g_directTStats.luFactorize;
                    g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                    if (g_sparseLuCache.solver.info() != Eigen::Success) {
                        writeLog(logFile, "--------疎直接法(DirectT): LU factorize failed on retry fallback");
                        return false;
                    }
                    g_sparseLuCache.factorized = true;
                }
                sol2 = g_sparseLuCache.solver.solve(b);
                if (g_sparseLuCache.solver.info() != Eigen::Success) {
                    writeLog(logFile, "--------疎直接法(DirectT): LU solve failed on retry");
                    return false;
                }
            }
            Eigen::VectorXd r2 = g_sparseLuCache.A * sol2 - b;
            const double maxResidual2 = (r2.size() > 0) ? r2.cwiseAbs().maxCoeff() : 0.0;
            if (!std::isfinite(maxResidual2)) {
                writeLog(logFile, "--------疎直接法(DirectT): LU retry residual is not finite");
                return false;
            }
            if (maxResidual2 > scaledTol) {
                std::ostringstream oss2;
                oss2 << "--------疎直接法(DirectT): LU retry residual still large: max|Ax-b|="
                     << std::scientific << std::setprecision(6) << maxResidual2
                     << " > tol=" << scaledTol;
                writeLog(logFile, oss2.str());
                return false;
            }
            ioSol = std::move(sol2);
            ioMethod = useKluBackend ? "KLU(fallback)" : "LU(fallback)";
            return true;
        };

        if (usedCholesky) {
            writeLog(logFile, "--------疎直接法(DirectT): retry with LU due to large residual after Cholesky");
            (void)tryLuFallback(sol, methodLabel);
        }

        std::ostringstream oss;
        oss << "--------疎直接法(DirectT): residual too large: max|Ax-b|="
            << std::scientific << std::setprecision(6) << maxResidual
            << " > tol=" << scaledTol
            << " (thermalTolerance=" << tolerance
            << ", bScale=" << bScale
            << ", method=" << methodLabel << ")";
        writeLog(logFile, oss.str());
        // LU fallback で methodLabel が置き換わっていれば成功しているので継続
        if (methodLabel == "LU(fallback)" || methodLabel == "KLU(fallback)") {
            // ok
        } else {
            return false;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) {
            writeLog(logFile, "--------疎直接法(DirectT): solution contains NaN/Inf");
            return false;
        }
        x[i] = v;
    }
    return true;
}

bool solveWithCachedFactorization(const Eigen::VectorXd& b,
                                  std::vector<double>& x,
                                  double tolerance,
                                  std::ostream& logFile,
                                  std::string& methodLabel) {
    const size_t n = x.size();
    if (n == 0) return true;
    bool useKluBackend = shouldUseKluBackend(logFile);

    Eigen::VectorXd sol;
    bool ok = false;
    static std::uint64_t s_cachedResidualCheckCounter = 0;

    if (g_cholCache.analyzed && g_cholCache.factorized && g_cholCache.patternSymmetric) {
        sol = g_cholCache.llt.solve(b);
        if (g_cholCache.llt.info() == Eigen::Success) {
            ok = true;
            methodLabel = "LLT(cached)";
        } else {
            sol = g_cholCache.ldlt.solve(b);
            if (g_cholCache.ldlt.info() == Eigen::Success) {
                ok = true;
                methodLabel = "LDLT(cached)";
            }
        }
    }
    if (!ok && g_sparseLuCache.analyzed && g_sparseLuCache.factorized) {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
        if (useKluBackend) {
            ok = solveWithKlu(b, sol, logFile);
            if (ok) {
                methodLabel = "KLU(cached)";
            } else {
                writeLog(logFile, "--------疎直接法(DirectT cached): fallback to Eigen::SparseLU after KLU(cached) solve failure");
                useKluBackend = false;
                g_sparseLuCache.factorized = false;
            }
        }
        if (!useKluBackend)
#endif
        {
            if (!g_sparseLuCache.factorized) {
                ++g_directTStats.luFactorize;
                g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                if (g_sparseLuCache.solver.info() != Eigen::Success) {
                    writeLog(logFile, "--------疎直接法(DirectT cached): LU factorize failed on fallback");
                    return false;
                }
                g_sparseLuCache.factorized = true;
            }
            sol = g_sparseLuCache.solver.solve(b);
            if (g_sparseLuCache.solver.info() == Eigen::Success) {
                ok = true;
                methodLabel = "LU(cached)";
            }
        }
    }
    if (!ok) return false;

    // cached 解の残差チェック:
    // 以前は間引いていたが、Cholesky系（LLT/LDLT）の cached 解が
    // timestep によって大残差になるケース（今回の再現）があるため、
    // LLT/LDLT の場合は毎回チェックして LU(cached) にフォールバックする。
    // LU(cached) の場合は従来どおり間引きでよい（性能優先）。
    const bool usedCholesky =
        (methodLabel.rfind("LLT", 0) == 0) || (methodLabel.rfind("LDLT", 0) == 0);
    const bool shouldCheckNow = usedCholesky || ((s_cachedResidualCheckCounter++ % 200) == 0);
    if (shouldCheckNow) {
        Eigen::VectorXd r = g_sparseLuCache.A * sol - b;
        const double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
        if (!std::isfinite(maxResidual)) {
            writeLog(logFile, "--------疎直接法(DirectT cached): residual is not finite");
            return false;
        }
        const double bScale = (b.size() > 0) ? b.cwiseAbs().maxCoeff() : 0.0;
        const double scaledTol = std::max(1.0, bScale) * tolerance * 10.0;
        if (maxResidual > scaledTol) {
            auto tryLuCachedFallback = [&](Eigen::VectorXd& ioSol, std::string& ioMethod) -> bool {
                if (!(g_sparseLuCache.analyzed && g_sparseLuCache.factorized)) return false;
                Eigen::VectorXd sol2;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
                if (useKluBackend) {
                    if (!solveWithKlu(b, sol2, logFile)) {
                        writeLog(logFile, "--------疎直接法(DirectT cached): fallback to Eigen::SparseLU after KLU(cached) retry failure");
                        useKluBackend = false;
                        g_sparseLuCache.factorized = false;
                    }
                }
                if (!useKluBackend)
#endif
                {
                    if (!g_sparseLuCache.factorized) {
                        ++g_directTStats.luFactorize;
                        g_sparseLuCache.solver.factorize(g_sparseLuCache.A);
                        if (g_sparseLuCache.solver.info() != Eigen::Success) {
                            writeLog(logFile, "--------疎直接法(DirectT cached): LU factorize failed on retry fallback");
                            return false;
                        }
                        g_sparseLuCache.factorized = true;
                    }
                    sol2 = g_sparseLuCache.solver.solve(b);
                    if (g_sparseLuCache.solver.info() != Eigen::Success) {
                        writeLog(logFile, "--------疎直接法(DirectT cached): LU(cached) solve failed on retry");
                        return false;
                    }
                }
                Eigen::VectorXd r2 = g_sparseLuCache.A * sol2 - b;
                const double maxResidual2 = (r2.size() > 0) ? r2.cwiseAbs().maxCoeff() : 0.0;
                if (!std::isfinite(maxResidual2)) {
                    writeLog(logFile, "--------疎直接法(DirectT cached): LU(cached) retry residual is not finite");
                    return false;
                }
                if (maxResidual2 > scaledTol) {
                    std::ostringstream oss2;
                    oss2 << "--------疎直接法(DirectT cached): LU(cached) retry residual still large: max|Ax-b|="
                         << std::scientific << std::setprecision(6) << maxResidual2
                         << " > tol=" << scaledTol;
                    writeLog(logFile, oss2.str());
                    return false;
                }
                ioSol = std::move(sol2);
                ioMethod = useKluBackend ? "KLU(cached-fallback)" : "LU(cached-fallback)";
                return true;
            };

            if (usedCholesky) {
                writeLog(logFile, "--------疎直接法(DirectT cached): retry with LU(cached) due to large residual after Cholesky");
                (void)tryLuCachedFallback(sol, methodLabel);
            }

            std::ostringstream oss;
            oss << "--------疎直接法(DirectT cached): residual too large: max|Ax-b|="
                << std::scientific << std::setprecision(6) << maxResidual
                << " > tol=" << scaledTol
                << " (thermalTolerance=" << tolerance
                << ", bScale=" << bScale
                << ", method=" << methodLabel << ")";
            writeLog(logFile, oss.str());
            if (methodLabel == "LU(cached-fallback)" || methodLabel == "KLU(cached-fallback)") {
                // ok
            } else {
                return false;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        const double v = sol[static_cast<int>(i)];
        if (!std::isfinite(v)) {
            writeLog(logFile, "--------疎直接法(DirectT cached): solution contains NaN/Inf");
            return false;
        }
        x[i] = v;
    }
    (void)logFile; // keep signature; log throttled by caller
    return true;
}

void resetOptionalDirectSolverCaches() {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    clearKluAll();
#endif
}

} // namespace ThermalSolverLinearDirect::detail


