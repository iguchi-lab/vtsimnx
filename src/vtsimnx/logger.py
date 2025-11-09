import logging
from datetime import datetime
from pathlib import Path

def get_logger(name: str = __name__) -> logging.Logger:
    # ログ保存先
    log_dir = Path.cwd()
    log_dir.mkdir(parents=True, exist_ok=True)

    # ログファイル
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"vtsimNx_{timestamp}.log"

    # ロガー作成
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # ファイルハンドラ（追加）
    fh = logging.FileHandler(log_path, encoding="utf-8-sig")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    return logger
