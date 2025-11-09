from .config_types import SimConfigType, IndexType, ToleranceType, CalcFlagType

from .logger import get_logger

logger = get_logger(__name__)

def parse(raw_config):
  logger.info("設定データの読み込み開始")
  try:

    # SimConfigTypeの型に従って初期化
    sim_config: SimConfigType = {
        "index": {"start": "", "end": "", "timestep": 0, "length": 0},
        "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6},
        "calc_flag": {"p": False, "t": False, "x": False, "c": False},
    }

    # 設定ファイルからシミュレーション設定を更新
    if "simulation" in raw_config:
        sim_data = raw_config["simulation"]
        # indexセクションの更新
        if "index" in sim_data:
            sim_config["index"].update(sim_data["index"])
        # toleranceセクションの更新
        if "tolerance" in sim_data:
            sim_config["tolerance"].update(sim_data["tolerance"])

    # 設定ファイルの保存
    output_json = {
        "simulation": sim_config,
        #"nodes": node_config,
        #"ventilation_branches": ventilation_config,
        #"thermal_branches": thermal_config,
        #"surfaces": surface_config,
        #"aircon": aircon_config,
    }

    logger.info("設定データの処理が完了しました")
    return output_json
  
  except Exception as e:
      logger.exception("エラーが発生しました: %s", e)
      raise
