from ./logger import get_logger

logger = get_logger(__name__)

def parse(input_data):
  logger.info("インプットデータの読み込み開始")
  print(input_data)
  return "get input_data"
