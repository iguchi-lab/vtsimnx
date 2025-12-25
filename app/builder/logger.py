import logging
from datetime import datetime
from pathlib import Path
import os


# 単一実行につき1つのログファイルに集約するため、
# 共通親ロガー（"vtsim_config"）へ一度だけ FileHandler を設定する。
APP_LOGGER_NAME = "vtsim_config"


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
    if parent.handlers and _handlers_usable(parent):
        return parent
    # 既存ハンドラが壊れている（ログdir削除など）場合は作り直す
    if parent.handlers and not _handlers_usable(parent):
        for h in list(parent.handlers):
            try:
                parent.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass
            except Exception:
                pass

    # デフォルトはリポジトリ直下の work/logs に出す（ルート汚染を避ける）
    repo_root = Path(__file__).resolve().parents[2]
    default_log_dir = repo_root / "work" / "logs"
    log_dir = Path(os.getenv("VTSIMNX_BUILDER_LOG_DIR", str(default_log_dir)))
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"vtsimNx_{timestamp}.log"

    parent.setLevel(logging.DEBUG)
    parent.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8-sig", delay=True)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    parent.addHandler(fh)
    return parent


def get_logger(name: str = __name__) -> logging.Logger:
    # 親ロガーへハンドラを一度だけ設定し、子ロガーを返す
    _ensure_parent_logger_initialized()
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")


