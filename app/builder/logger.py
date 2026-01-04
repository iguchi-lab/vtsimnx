import logging
from datetime import datetime
from pathlib import Path
import os
import threading
from contextlib import contextmanager


# 単一実行につき1つのログファイルに集約するため、
# 共通親ロガー（"vtsim_config"）へ一度だけ FileHandler を設定する。
APP_LOGGER_NAME = "vtsim_config"
_CONFIG_LOCK = threading.RLock()


def _handlers_usable(parent: logging.Logger) -> bool:
    """
    既存ハンドラが「今もファイルへ出力できる状態」かを判定する。

    よくある事故:
    - APIプロセス起動中に work/ や work/logs を rm -rf した場合、
      FileHandler は削除済みファイル（unlinkされた inode）へ書き続け、
      ディレクトリ上はログが「無い」ように見える。

    対策:
    - ログディレクトリが存在しない場合は再初期化
    - 既に stream を開いているのに、baseFilename が存在しない場合は再初期化
      （= unlink された可能性が高い）
    """
    for h in parent.handlers:
        if isinstance(h, logging.FileHandler):
            base = getattr(h, "baseFilename", None)
            if not base:
                continue
            p = Path(str(base))
            # ディレクトリが消えている: そのままでは復旧不能なので作り直す
            if not p.parent.exists():
                return False
            # 既にファイルを開いているのにパスが無い: unlink されている可能性が高い
            if getattr(h, "stream", None) is not None and not p.exists():
                return False
    return True


def _ensure_parent_logger_initialized() -> logging.Logger:
    parent = logging.getLogger(APP_LOGGER_NAME)

    # 要求された出力先（env）:
    # - VTSIMNX_BUILDER_LOG_FILE: 単一ファイルへ固定で出力（APIの1リクエスト=1ログにしたい用途）
    # - VTSIMNX_BUILDER_LOG_DIR : 指定ディレクトリ配下へ出力（ファイル名は自動）
    requested_file = os.getenv("VTSIMNX_BUILDER_LOG_FILE")
    requested_dir = os.getenv("VTSIMNX_BUILDER_LOG_DIR")

    def _current_filehandler_path() -> Path | None:
        for h in parent.handlers:
            if isinstance(h, logging.FileHandler):
                base = getattr(h, "baseFilename", None)
                if base:
                    try:
                        return Path(str(base)).resolve()
                    except Exception:
                        return Path(str(base))
        return None

    def _reset_handlers() -> None:
        for h in list(parent.handlers):
            try:
                parent.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            except Exception:
                pass

    if parent.handlers:
        usable = _handlers_usable(parent)
        if usable:
            # 既存ハンドラが健全でも、「出力先が変わった」場合は確実に付け替える。
            cur = _current_filehandler_path()
            if requested_file:
                try:
                    want = Path(requested_file).resolve()
                except Exception:
                    want = Path(requested_file)
                if cur is not None and cur == want:
                    return parent
                _reset_handlers()
            elif requested_dir:
                try:
                    want_dir = Path(requested_dir).resolve()
                except Exception:
                    want_dir = Path(requested_dir)
                if cur is not None and cur.parent == want_dir:
                    return parent
                _reset_handlers()
            else:
                return parent
        else:
            # 既存ハンドラが壊れている（ログdir削除など）場合は作り直す
            _reset_handlers()

    # 出力先の決定
    if requested_file:
        log_path = Path(requested_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # デフォルトはリポジトリ直下の work/logs に出す（ルート汚染を避ける）
        repo_root = Path(__file__).resolve().parents[2]
        default_log_dir = repo_root / "work" / "logs"
        log_dir = Path(os.getenv("VTSIMNX_BUILDER_LOG_DIR", str(default_log_dir)))
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"vtsimNx_{timestamp}.log"

    parent.setLevel(logging.DEBUG)
    parent.propagate = False

    # delay=False: 初回出力までの間にディレクトリが消える等のレースで Errno2 になりやすいので、ここで確実に open する
    fh = logging.FileHandler(log_path, encoding="utf-8-sig", delay=False)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    parent.addHandler(fh)
    return parent


def get_logger(name: str = __name__) -> logging.Logger:
    # 親ロガーへハンドラを一度だけ設定し、子ロガーを返す
    _ensure_parent_logger_initialized()
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")


def _reset_parent_handlers() -> None:
    parent = logging.getLogger(APP_LOGGER_NAME)
    for h in list(parent.handlers):
        try:
            parent.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        except Exception:
            pass


@contextmanager
def use_builder_log_file(log_file: Path):
    """
    builder のログ出力先を「このブロックの間だけ」単一ファイルへ固定する。
    - FastAPI の並列リクエストでログが混ざらないよう、ロガー設定操作をロックして直列化する。
    """
    with _CONFIG_LOCK:
        prev = os.environ.get("VTSIMNX_BUILDER_LOG_FILE")
        os.environ["VTSIMNX_BUILDER_LOG_FILE"] = str(log_file)
        try:
            # 出力先を確実に切り替える
            _reset_parent_handlers()
            _ensure_parent_logger_initialized()
            yield
        finally:
            # このリクエストのログに書き続けないよう、ハンドラは必ず外す
            _reset_parent_handlers()
            if prev is None:
                os.environ.pop("VTSIMNX_BUILDER_LOG_FILE", None)
            else:
                os.environ["VTSIMNX_BUILDER_LOG_FILE"] = prev


