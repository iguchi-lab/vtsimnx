import logging
from datetime import datetime
from pathlib import Path


# 単一実行につき1つのログファイルに集約するため、
# 共通親ロガー（"vtsimnx"）へ一度だけ FileHandler を設定する。
APP_LOGGER_NAME = "vtsimnx"


def _ensure_parent_logger_initialized() -> logging.Logger:
    parent = logging.getLogger(APP_LOGGER_NAME)
    if parent.handlers:
        return parent

    log_dir = Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"vtsimNx_{timestamp}.log"

    parent.setLevel(logging.DEBUG)
    parent.propagate = False

    fh = logging.FileHandler(log_path, encoding="utf-8-sig")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    parent.addHandler(fh)
    return parent


def get_logger(name: str = __name__) -> logging.Logger:
    # 親ロガーへハンドラを一度だけ設定し、子ロガーを返す
    _ensure_parent_logger_initialized()
    return logging.getLogger(f"{APP_LOGGER_NAME}.{name}")
