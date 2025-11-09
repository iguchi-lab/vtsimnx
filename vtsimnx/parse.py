from .logger import get_logger

logger = get_logger(__name__)

def parse(input_data):
  logger.info("設定データの読み込み開始")
  logger.info("設定データの処理が完了しました")
  return 0
