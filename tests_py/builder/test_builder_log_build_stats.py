"""
ビルド結果（ノード数・ブランチ数）が build_stats_out に渡り、
attach_builder_log_to_artifacts で builder.log に追記されることを軽い設定で検証する。
重いビルドなしで実行できる。
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from app.builder import build_config_with_warning_details
from app.builder.logger import use_builder_log_file
from app.solver_runner import attach_builder_log_to_artifacts


# 軽い設定（表面なし・ノード1・換気0・熱0 → バリデを通す最小構成）
MINIMAL_CONFIG = {
    "simulation": {
        "index": {
            "start": "2025-01-01T00:00:00Z",
            "end": "2025-01-01T00:01:00Z",
            "timestep": 60,
            "length": 2,
        },
        "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
    },
    "nodes": [{"key": "LD", "t": 20.0, "calc_t": True}],
    "ventilation_branches": [],
    "thermal_branches": [],
}


def test_build_stats_out_is_filled():
    """build_config_with_warning_details に build_stats_out=[] を渡すと中身が (ノード数, 熱ブランチ数, 換気ブランチ数) で埋まる。"""
    build_stats_out: list = []
    built, warnings, _ = build_config_with_warning_details(
        MINIMAL_CONFIG, output_path=None, build_stats_out=build_stats_out
    )
    assert len(build_stats_out) == 1
    n_nodes, n_thermal, n_vent = build_stats_out[0]
    assert isinstance(n_nodes, int) and n_nodes >= 0
    assert isinstance(n_thermal, int) and n_thermal >= 0
    assert isinstance(n_vent, int) and n_vent >= 0
    assert n_nodes == len(built.get("nodes") or [])
    assert n_thermal == len(built.get("thermal_branches") or [])
    assert n_vent == len(built.get("ventilation_branches") or [])


def test_attach_builder_log_appends_build_result_line(tmp_path: Path):
    """attach_builder_log_to_artifacts に build_stats を渡すと、コピー先の builder.log に「ビルド結果」行が含まれる。"""
    # work 配下に artifact_dir を作る必要がある（attach は BASE_DIR/work を参照）
    work_dir = Path(__file__).resolve().parents[2] / "work"
    artifact_dir_name = "output.test_build_stats.artifacts.1"
    artifact_dir = work_dir / artifact_dir_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 元の builder ログ（1行だけ）
    builder_log_tmp = tmp_path / "builder.log"
    builder_log_tmp.write_text("2025-01-01 00:00:00,000 [INFO] 設定データの読み込み開始\n", encoding="utf-8")

    output_data = {"artifact_dir": artifact_dir_name, "index": {"length": 1}}

    build_stats = (2, 3, 1)  # ノード2, 熱ブランチ3, 換気1
    dest = attach_builder_log_to_artifacts(
        output_data,
        builder_log_path=builder_log_tmp,
        artifact_filename="builder.log",
        delete_source=False,
        build_stats=build_stats,
    )
    assert dest is not None
    content = dest.read_text(encoding="utf-8")
    assert "ビルド結果: ノード=2, 熱ブランチ=3, 換気ブランチ=1" in content
    assert "設定データの読み込み開始" in content
