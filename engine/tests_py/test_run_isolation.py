"""run_id 作業ディレクトリ隔離のテスト。"""
import threading
from pathlib import Path

import pytest

from app.solver_runner import cleanup_run_workdir, run_workdir


def test_cleanup_run_workdir_does_not_touch_other_runs(tmp_path, monkeypatch):
    import app.solver_runner as sr

    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

    run_a = run_workdir("aaa111")
    run_b = run_workdir("bbb222")
    (run_a / "input.aaa111.json").write_text("{}", encoding="utf-8")
    (run_a / "output.aaa111.json").write_text("{}", encoding="utf-8")
    (run_a / "artifacts.keep.output.aaa111").mkdir()
    (run_a / "artifacts.keep.output.aaa111" / "x.txt").write_text("keep", encoding="utf-8")
    (run_b / "input.bbb222.json").write_text("{}", encoding="utf-8")
    (run_b / "builder.log.tmp").write_text("log", encoding="utf-8")

    cleanup_run_workdir("aaa111", keep_artifacts=True)

    assert not (run_a / "input.aaa111.json").exists()
    assert not (run_a / "output.aaa111.json").exists()
    assert (run_a / "artifacts.keep.output.aaa111" / "x.txt").exists()
    assert (run_b / "input.bbb222.json").exists()
    assert (run_b / "builder.log.tmp").exists()


def test_parallel_cleanup_isolation(tmp_path, monkeypatch):
    import app.solver_runner as sr

    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)

    ids = [f"run{i:04d}" for i in range(8)]
    for rid in ids:
        d = run_workdir(rid)
        (d / f"input.{rid}.json").write_text("{}", encoding="utf-8")
        (d / f"artifacts.{rid}").mkdir()
        (d / f"artifacts.{rid}" / "ok.txt").write_text("1", encoding="utf-8")

    errors: list[str] = []

    def worker(rid: str) -> None:
        try:
            cleanup_run_workdir(rid, keep_artifacts=True)
            d = tmp_path / "work" / "runs" / rid
            if (d / f"input.{rid}.json").exists():
                errors.append(f"{rid}: input still exists")
            if not (d / f"artifacts.{rid}" / "ok.txt").exists():
                errors.append(f"{rid}: artifact missing")
            # 他 run の artifact が消えていないこと
            for other in ids:
                if other == rid:
                    continue
                other_art = tmp_path / "work" / "runs" / other / f"artifacts.{other}" / "ok.txt"
                if not other_art.exists():
                    errors.append(f"{rid} wiped {other}")
        except Exception as e:
            errors.append(f"{rid}: {e}")

    threads = [threading.Thread(target=worker, args=(rid,)) for rid in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []


def test_resolve_artifact_path_searches_runs(tmp_path, monkeypatch):
    import app.solver_runner as sr

    monkeypatch.setattr(sr, "BASE_DIR", tmp_path)
    run_dir = run_workdir("abc")
    art = run_dir / "artifacts.demo"
    art.mkdir()
    (art / "f.txt").write_text("x", encoding="utf-8")

    found = sr.resolve_artifact_path("artifacts.demo")
    assert found is not None
    assert found == art.resolve()

    # 直下優先
    direct = tmp_path / "work" / "artifacts.demo"
    direct.mkdir()
    found2 = sr.resolve_artifact_path("artifacts.demo")
    assert found2 == direct.resolve()
