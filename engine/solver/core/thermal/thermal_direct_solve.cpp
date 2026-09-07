#include "core/thermal/thermal_direct_internal.h"

#include <cstdlib>

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
#include <klu.h>
#endif

namespace ThermalSolverLinearDirect::detail {

namespace {

constexpr const char* kLuBackendEnv = "VTSIMNX_THERMAL_DIRECT_LU";

/** KLU 経路では Eigen SparseLU の analyze を飛ばすため、フォールバック前に必須。 */
inline void ensureEigenSparseLuAnalyzed(SparseLUCache& sparseLu) {
    if (sparseLu.eigenAnalyzed) return;
    sparseLu.solver.analyzePattern(sparseLu.A);
    sparseLu.eigenAnalyzed = true;
}

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
void clearKluNumeric(KluCache& klu) {
    if (klu.numeric != nullptr) {
        klu_free_numeric(&klu.numeric, &klu.common);
        klu.numeric = nullptr;
    }
    klu.factorized = false;
}

void clearKluAll(KluCache& klu) {
    clearKluNumeric(klu);
    if (klu.symbolic != nullptr) {
        klu_free_symbolic(&klu.symbolic, &klu.common);
        klu.symbolic = nullptr;
    }
    klu.analyzed = false;
    klu.n = 0;
    klu.nnz = 0;
    klu.patternHash = 0;
    klu.valueHash = 0;
    klu_defaults(&klu.common);
}
#endif

bool shouldUseKluBackend(DirectTSolverContext& ctx, std::ostream& logFile) {
    if (ctx.luBackendState >= 0) return ctx.luBackendState == 1;

    const char* env = std::getenv(kLuBackendEnv);
    const std::string requested = (env != nullptr) ? std::string(env) : std::string();
    const bool forceLu = (requested == "lu" || requested == "LU" ||
                          requested == "eigen" || requested == "EIGEN" ||
                          requested == "sparselu" || requested == "SPARSELU");
    const bool forceKlu = (requested == "klu" || requested == "KLU");
    if (forceLu) {
        ctx.luBackendState = 0;
        writeDomainLog(logFile, "熱", "[INFO] DirectT: LU backend=Eigen::SparseLU (forced by env)");
        return false;
    }

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    if (!requested.empty() && !forceKlu) {
        ctx.luBackendState = 1;
        writeDomainLog(logFile, "熱", "[WARN] DirectT: 不明な LU backend env。既定 KLU を使用");
        return true;
    }
    ctx.luBackendState = 1;
    writeDomainLog(logFile, "熱", "[INFO] DirectT: LU backend=KLU");
    return true;
#else
    ctx.luBackendState = 0;
    if (forceKlu) {
        writeDomainLog(logFile, "熱", "[WARN] DirectT: KLU 不可のため Eigen::SparseLU へフォールバック");
    } else {
        writeDomainLog(logFile, "熱", "[INFO] DirectT: LU backend=Eigen::SparseLU (KLU なし)");
    }
    return false;
#endif
}

#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
bool ensureKluPattern(KluCache& klu, const Eigen::SparseMatrix<double>& A, int n, size_t nnz, std::uint64_t patternHash, std::ostream& logFile) {
    const bool needAnalyze = (!klu.analyzed) ||
                             (klu.n != n) ||
                             (klu.nnz != nnz) ||
                             (klu.patternHash != patternHash);
    if (!needAnalyze) return true;

    clearKluAll(klu);
    klu.n = n;
    klu.nnz = nnz;
    klu.patternHash = patternHash;
    klu.symbolic = klu_analyze(n, const_cast<int*>(A.outerIndexPtr()), const_cast<int*>(A.innerIndexPtr()), &klu.common);
    if (klu.symbolic == nullptr) {
        writeDomainLog(logFile, "熱", "[ERROR] DirectT: KLU analyze に失敗");
        return false;
    }
    klu.analyzed = true;
    return true;
}

bool factorizeWithKlu(KluCache& klu, const Eigen::SparseMatrix<double>& A, std::uint64_t valueHash, std::ostream& logFile) {
    if (!klu.analyzed) return false;
    if (klu.factorized && klu.valueHash == valueHash) return true;

    clearKluNumeric(klu);
    klu.numeric = klu_factor(const_cast<int*>(A.outerIndexPtr()),
                             const_cast<int*>(A.innerIndexPtr()),
                             const_cast<double*>(A.valuePtr()),
                             klu.symbolic,
                             &klu.common);
    if (klu.numeric == nullptr) {
        writeDomainLog(logFile, "熱", "[ERROR] DirectT: KLU 因子分解に失敗（特異/悪条件）");
        return false;
    }
    klu.factorized = true;
    klu.valueHash = valueHash;
    return true;
}

bool solveWithKlu(KluCache& klu, const Eigen::VectorXd& b, Eigen::VectorXd& sol, std::ostream& logFile) {
    if (!klu.analyzed || !klu.factorized) return false;
    sol = b;
    const int ok = klu_solve(klu.symbolic, klu.numeric, klu.n, 1, sol.data(), &klu.common);
    if (ok == 0) {
        writeDomainLog(logFile, "熱", "[ERROR] DirectT: KLU solve に失敗");
        return false;
    }
    return true;
}
#endif

} // namespace

bool solveSparseDirect(DirectTSolverContext& ctx,
                       const LinearSystem& system,
                       std::vector<double>& x,
                       double tolerance,
                       std::ostream& logFile,
                       std::string& methodLabel) {
    using thermal_linear_utils::fnv1a64_update;
    using thermal_linear_utils::hashDoubleBits;
    using thermal_linear_utils::isSymmetricPatternByCols;

    auto& sparseLu = ctx.sparseLu;
    auto& chol = ctx.chol;
    auto& stats = ctx.stats;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    auto& klu = ctx.klu;
#endif

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
    bool useKluBackend = shouldUseKluBackend(ctx, logFile);

    const bool needRebuildPattern = (!sparseLu.analyzed) ||
                                   (sparseLu.n != static_cast<int>(n)) ||
                                   (sparseLu.nnz != nnz) ||
                                   (sparseLu.patternHash != patternHash);

    if (needRebuildPattern) {
        ++stats.patternRebuild;
        sparseLu.analyzed = false;
        sparseLu.eigenAnalyzed = false;
        sparseLu.n = static_cast<int>(n);
        sparseLu.nnz = nnz;
        sparseLu.patternHash = patternHash;
        sparseLu.factorized = false;
        sparseLu.valueHash = 0;
        sparseLu.valuePtrIndexByRow.clear();
        sparseLu.A.resize(0, 0);

        sparseLu.solver.~SparseLU();
        new (&sparseLu.solver) Eigen::SparseLU<Eigen::SparseMatrix<double>>();

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
        sparseLu.A = Eigen::SparseMatrix<double>(static_cast<int>(n), static_cast<int>(n));
        sparseLu.A.setFromTriplets(triplets.begin(), triplets.end());
        sparseLu.A.makeCompressed();

        // valuePtr mapping (row-wise)
        sparseLu.valuePtrIndexByRow.assign(n, {});
        std::vector<std::vector<std::pair<int, int>>> rowEntries(n);
        for (size_t r = 0; r < n; ++r) rowEntries[r].reserve(system.colIndices[r].size());
        double* base = sparseLu.A.valuePtr();
        for (int outer = 0; outer < sparseLu.A.outerSize(); ++outer) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(sparseLu.A, outer); it; ++it) {
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
            sparseLu.valuePtrIndexByRow[r].assign(cols.size(), -1);
            size_t j = 0;
            for (size_t k = 0; k < cols.size(); ++k) {
                const int col = cols[k];
                while (j < entries.size() && entries[j].first < col) ++j;
                if (j < entries.size() && entries[j].first == col) sparseLu.valuePtrIndexByRow[r][k] = entries[j].second;
            }
        }
        bool mappingOk = true;
        for (size_t r = 0; r < n && mappingOk; ++r) {
            for (int p : sparseLu.valuePtrIndexByRow[r]) {
                if (p < 0) { mappingOk = false; break; }
            }
        }
        if (!mappingOk) {
            writeDomainLog(logFile, "熱", "[ERROR] DirectT: valuePtrIndexByRow の構築に失敗（パターン不一致）。停止します。");
            sparseLu.analyzed = false;
            sparseLu.eigenAnalyzed = false;
            sparseLu.factorized = false;
            sparseLu.valueHash = 0;
            sparseLu.valuePtrIndexByRow.clear();
            chol.analyzed = false;
            chol.factorized = false;
            return false;
        }

        if (useKluBackend) {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (!ensureKluPattern(klu, sparseLu.A, static_cast<int>(n), nnz, patternHash, logFile)) {
                writeDomainLog(logFile, "熱", "[WARN] DirectT: fallback to Eigen::SparseLU after KLU analyze failure");
                useKluBackend = false;
            }
            if (useKluBackend) {
                sparseLu.analyzed = true;
                sparseLu.eigenAnalyzed = false;
            } else {
                sparseLu.solver.analyzePattern(sparseLu.A);
                sparseLu.analyzed = true;
                sparseLu.eigenAnalyzed = true;
            }
#else
            sparseLu.analyzed = false;
            sparseLu.eigenAnalyzed = false;
            return false;
#endif
        } else {
            sparseLu.solver.analyzePattern(sparseLu.A);
            sparseLu.analyzed = true;
            sparseLu.eigenAnalyzed = true;
        }
        sparseLu.valueHash = valueHash;

        chol.analyzed = false;
        chol.factorized = false;
        chol.patternSymmetric = isSymmetricPatternByCols(system.colIndices);
    } else {
        std::uint64_t valueHash = 0;
        for (size_t i = 0; i < n; ++i) {
            const auto& vals = system.A[i];
            for (size_t k = 0; k < vals.size(); ++k) {
                const int p = sparseLu.valuePtrIndexByRow[i][k];
                if (p < 0) return false;
                sparseLu.A.valuePtr()[p] = vals[k];
                valueHash = hashDoubleBits(valueHash, vals[k]);
            }
        }
        if (sparseLu.valueHash != valueHash) {
            sparseLu.factorized = false;
            sparseLu.valueHash = valueHash;
            chol.factorized = false;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            klu.factorized = false;
#endif
        }
    }

    const bool symmetricCandidate = chol.patternSymmetric;
    Eigen::VectorXd sol;
    bool solved = false;

    if (symmetricCandidate) {
        const bool needAnalyze = (!chol.analyzed) ||
                                 (chol.n != static_cast<int>(n)) ||
                                 (chol.nnz != nnz) ||
                                 (chol.patternHash != patternHash);
        if (needAnalyze) {
            chol.analyzed = false;
            chol.n = static_cast<int>(n);
            chol.nnz = nnz;
            chol.patternHash = patternHash;
            chol.factorized = false;
            chol.valueHash = 0;
            chol.llt.analyzePattern(sparseLu.A);
            chol.ldlt.analyzePattern(sparseLu.A);
            chol.analyzed = true;
        }

        if (!chol.factorized || chol.valueHash != sparseLu.valueHash) {
            ++stats.cholFactorize;
            chol.llt.factorize(sparseLu.A);
            if (chol.llt.info() == Eigen::Success) {
                chol.factorized = true;
                chol.valueHash = sparseLu.valueHash;
                sol = chol.llt.solve(b);
                if (chol.llt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LLT";
                }
            } else {
                writeDomainLog(logFile, "熱", "[ERROR] DirectT: LLT 因子分解に失敗（非SPD/悪条件）");
            }
            if (!solved) {
                ++stats.cholFactorize;
                chol.ldlt.factorize(sparseLu.A);
                if (chol.ldlt.info() == Eigen::Success) {
                    chol.factorized = true;
                    chol.valueHash = sparseLu.valueHash;
                    sol = chol.ldlt.solve(b);
                    if (chol.ldlt.info() == Eigen::Success) {
                        solved = true;
                        methodLabel = "LDLT";
                    }
                } else {
                    writeDomainLog(logFile, "熱", "[ERROR] DirectT: LDLT 因子分解に失敗（非SPD/不定/悪条件）");
                }
            }
            if (!solved) {
                chol.factorized = false;
                chol.patternSymmetric = false;
            }
        } else {
            sol = chol.llt.solve(b);
            if (chol.llt.info() == Eigen::Success) {
                solved = true;
                methodLabel = "LLT(cached)";
            } else {
                sol = chol.ldlt.solve(b);
                if (chol.ldlt.info() == Eigen::Success) {
                    solved = true;
                    methodLabel = "LDLT(cached)";
                }
            }
        }
    }

    if (!solved) {
        if (!sparseLu.factorized) {
            ++stats.luFactorize;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (useKluBackend) {
                if (!factorizeWithKlu(klu, sparseLu.A, sparseLu.valueHash, logFile)) {
                    writeDomainLog(logFile, "熱", "[WARN] DirectT: fallback to Eigen::SparseLU after KLU factorize failure");
                    useKluBackend = false;
                }
            }
            if (!useKluBackend)
#endif
            {
                ensureEigenSparseLuAnalyzed(sparseLu);
                sparseLu.solver.factorize(sparseLu.A);
                if (sparseLu.solver.info() != Eigen::Success) {
                    writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU 因子分解に失敗（特異/悪条件）");
                    return false;
                }
            }
            sparseLu.factorized = true;
        }
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
        if (useKluBackend) {
            if (solveWithKlu(klu, b, sol, logFile)) {
                methodLabel = "KLU";
            } else {
                writeDomainLog(logFile, "熱", "[WARN] DirectT: fallback to Eigen::SparseLU after KLU solve failure");
                useKluBackend = false;
                sparseLu.factorized = false;
            }
        }
        if (!useKluBackend)
#endif
        {
            if (!sparseLu.factorized) {
                ++stats.luFactorize;
                ensureEigenSparseLuAnalyzed(sparseLu);
                sparseLu.solver.factorize(sparseLu.A);
                if (sparseLu.solver.info() != Eigen::Success) {
                    writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU 因子分解に失敗（フォールバック経路）");
                    return false;
                }
                sparseLu.factorized = true;
            }
            sol = sparseLu.solver.solve(b);
            if (sparseLu.solver.info() != Eigen::Success) {
                writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU solve に失敗");
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
    Eigen::VectorXd r = sparseLu.A * sol - b;
    const double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
    if (!std::isfinite(maxResidual)) {
        writeDomainLog(logFile, "熱", "[ERROR] DirectT: 残差が非有限");
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
            if (!sparseLu.factorized) {
                ++stats.luFactorize;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
                if (useKluBackend) {
                    if (!factorizeWithKlu(klu, sparseLu.A, sparseLu.valueHash, logFile)) {
                        writeDomainLog(logFile, "熱", "[WARN] DirectT: fallback to Eigen::SparseLU after KLU factorize retry failure");
                        useKluBackend = false;
                    }
                }
                if (!useKluBackend)
#endif
                {
                    ensureEigenSparseLuAnalyzed(sparseLu);
                    sparseLu.solver.factorize(sparseLu.A);
                    if (sparseLu.solver.info() != Eigen::Success) {
                        writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU 因子分解に失敗（再試行）");
                        return false;
                    }
                }
                sparseLu.factorized = true;
            }
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
            if (useKluBackend) {
                if (!solveWithKlu(klu, b, sol2, logFile)) {
                    writeDomainLog(logFile, "熱", "[WARN] DirectT: fallback to Eigen::SparseLU after KLU solve retry failure");
                    useKluBackend = false;
                    sparseLu.factorized = false;
                }
            }
            if (!useKluBackend)
#endif
            {
                if (!sparseLu.factorized) {
                    ++stats.luFactorize;
                    ensureEigenSparseLuAnalyzed(sparseLu);
                    sparseLu.solver.factorize(sparseLu.A);
                    if (sparseLu.solver.info() != Eigen::Success) {
                        writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU 因子分解に失敗（再試行フォールバック）");
                        return false;
                    }
                    sparseLu.factorized = true;
                }
                sol2 = sparseLu.solver.solve(b);
                if (sparseLu.solver.info() != Eigen::Success) {
                    writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU solve に失敗（再試行）");
                    return false;
                }
            }
            Eigen::VectorXd r2 = sparseLu.A * sol2 - b;
            const double maxResidual2 = (r2.size() > 0) ? r2.cwiseAbs().maxCoeff() : 0.0;
            if (!std::isfinite(maxResidual2)) {
                writeDomainLog(logFile, "熱", "[ERROR] DirectT: LU 再試行残差が非有限");
                return false;
            }
            if (maxResidual2 > scaledTol) {
                std::ostringstream oss2;
                oss2 << "--------疎直接法(DirectT): LU retry residual still large: max|Ax-b|="
                     << std::scientific << std::setprecision(6) << maxResidual2
                     << " > tol=" << scaledTol;
                writeDomainLog(logFile, "熱", oss2.str());
                return false;
            }
            ioSol = std::move(sol2);
            ioMethod = useKluBackend ? "KLU(fallback)" : "LU(fallback)";
            return true;
        };

        if (usedCholesky) {
            writeDomainLog(logFile, "熱", "[WARN] DirectT: retry with LU due to large residual after Cholesky");
            (void)tryLuFallback(sol, methodLabel);
        }

        std::ostringstream oss;
        oss << "[ERROR] DirectT: residual too large: max|Ax-b|="
            << std::scientific << std::setprecision(6) << maxResidual
            << " > tol=" << scaledTol
            << " (thermalLinearResidualRelativeTolerance=" << tolerance
            << ", bScale=" << bScale
            << ", method=" << methodLabel << ")";
        writeDomainLog(logFile, "熱", oss.str());
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
            writeDomainLog(logFile, "熱", "[ERROR] DirectT: 解に NaN/Inf を含む");
            return false;
        }
        x[i] = v;
    }
    return true;
}

bool solveWithCachedFactorization(DirectTSolverContext& ctx,
                                  const Eigen::VectorXd& b,
                                  std::vector<double>& x,
                                  double tolerance,
                                  std::ostream& logFile,
                                  std::string& methodLabel) {
    auto& sparseLu = ctx.sparseLu;
    auto& chol = ctx.chol;
    auto& stats = ctx.stats;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    auto& klu = ctx.klu;
#endif

    const size_t n = x.size();
    if (n == 0) return true;
    bool useKluBackend = shouldUseKluBackend(ctx, logFile);

    Eigen::VectorXd sol;
    bool ok = false;

    if (chol.analyzed && chol.factorized && chol.patternSymmetric) {
        sol = chol.llt.solve(b);
        if (chol.llt.info() == Eigen::Success) {
            ok = true;
            methodLabel = "LLT(cached)";
        } else {
            sol = chol.ldlt.solve(b);
            if (chol.ldlt.info() == Eigen::Success) {
                ok = true;
                methodLabel = "LDLT(cached)";
            }
        }
    }
    if (!ok && sparseLu.analyzed && sparseLu.factorized) {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
        if (useKluBackend) {
            ok = solveWithKlu(klu, b, sol, logFile);
            if (ok) {
                methodLabel = "KLU(cached)";
            } else {
                writeDomainLog(logFile, "熱", "[WARN] DirectT cached: fallback to Eigen::SparseLU after KLU(cached) solve failure");
                useKluBackend = false;
                sparseLu.factorized = false;
            }
        }
        if (!useKluBackend)
#endif
        {
            if (!sparseLu.factorized) {
                ++stats.luFactorize;
                ensureEigenSparseLuAnalyzed(sparseLu);
                sparseLu.solver.factorize(sparseLu.A);
                if (sparseLu.solver.info() != Eigen::Success) {
                    writeDomainLog(logFile, "熱", "[WARN] DirectT cached: LU factorize failed on fallback");
                    return false;
                }
                sparseLu.factorized = true;
            }
            sol = sparseLu.solver.solve(b);
            if (sparseLu.solver.info() == Eigen::Success) {
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
    const bool shouldCheckNow = usedCholesky || ((ctx.cachedResidualCheckCounter++ % 200) == 0);
    if (shouldCheckNow) {
        Eigen::VectorXd r = sparseLu.A * sol - b;
        const double maxResidual = (r.size() > 0) ? r.cwiseAbs().maxCoeff() : 0.0;
        if (!std::isfinite(maxResidual)) {
            writeDomainLog(logFile, "熱", "[WARN] DirectT cached: residual is not finite");
            return false;
        }
        const double bScale = (b.size() > 0) ? b.cwiseAbs().maxCoeff() : 0.0;
        const double scaledTol = std::max(1.0, bScale) * tolerance * 10.0;
        if (maxResidual > scaledTol) {
            auto tryLuCachedFallback = [&](Eigen::VectorXd& ioSol, std::string& ioMethod) -> bool {
                if (!(sparseLu.analyzed && sparseLu.factorized)) return false;
                Eigen::VectorXd sol2;
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
                if (useKluBackend) {
                    if (!solveWithKlu(klu, b, sol2, logFile)) {
                        writeDomainLog(logFile, "熱", "[WARN] DirectT cached: fallback to Eigen::SparseLU after KLU(cached) retry failure");
                        useKluBackend = false;
                        sparseLu.factorized = false;
                    }
                }
                if (!useKluBackend)
#endif
                {
                    if (!sparseLu.factorized) {
                        ++stats.luFactorize;
                        ensureEigenSparseLuAnalyzed(sparseLu);
                        sparseLu.solver.factorize(sparseLu.A);
                        if (sparseLu.solver.info() != Eigen::Success) {
                            writeDomainLog(logFile, "熱", "[WARN] DirectT cached: LU factorize failed on retry fallback");
                            return false;
                        }
                        sparseLu.factorized = true;
                    }
                    sol2 = sparseLu.solver.solve(b);
                    if (sparseLu.solver.info() != Eigen::Success) {
                        writeDomainLog(logFile, "熱", "[WARN] DirectT cached: LU(cached) solve failed on retry");
                        return false;
                    }
                }
                Eigen::VectorXd r2 = sparseLu.A * sol2 - b;
                const double maxResidual2 = (r2.size() > 0) ? r2.cwiseAbs().maxCoeff() : 0.0;
                if (!std::isfinite(maxResidual2)) {
                    writeDomainLog(logFile, "熱", "[WARN] DirectT cached: LU(cached) retry residual is not finite");
                    return false;
                }
                if (maxResidual2 > scaledTol) {
                    std::ostringstream oss2;
                    oss2 << "--------疎直接法(DirectT cached): LU(cached) retry residual still large: max|Ax-b|="
                         << std::scientific << std::setprecision(6) << maxResidual2
                         << " > tol=" << scaledTol;
                    writeDomainLog(logFile, "熱", oss2.str());
                    return false;
                }
                ioSol = std::move(sol2);
                ioMethod = useKluBackend ? "KLU(cached-fallback)" : "LU(cached-fallback)";
                return true;
            };

            if (usedCholesky) {
                writeDomainLog(logFile, "熱", "[WARN] DirectT cached: retry with LU(cached) due to large residual after Cholesky");
                (void)tryLuCachedFallback(sol, methodLabel);
            }

            std::ostringstream oss;
            oss << "[ERROR] DirectT cached: residual too large: max|Ax-b|="
                << std::scientific << std::setprecision(6) << maxResidual
                << " > tol=" << scaledTol
                << " (thermalLinearResidualRelativeTolerance=" << tolerance
                << ", bScale=" << bScale
                << ", method=" << methodLabel << ")";
            writeDomainLog(logFile, "熱", oss.str());
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
            writeDomainLog(logFile, "熱", "[WARN] DirectT cached: solution contains NaN/Inf");
            return false;
        }
        x[i] = v;
    }
    (void)logFile; // keep signature; log throttled by caller
    return true;
}

void resetOptionalDirectSolverCaches(DirectTSolverContext& ctx) {
#if defined(VTSIMNX_USE_KLU) && (VTSIMNX_USE_KLU)
    clearKluAll(ctx.klu);
#else
    (void)ctx;
#endif
}

} // namespace ThermalSolverLinearDirect::detail


