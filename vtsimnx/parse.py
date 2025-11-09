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
        # データソースのデータをノード、換気ブランチ、熱ブランチ、表面、材料、層に設定
        for i, node in enumerate(raw_config["nodes"]):
            for k, v in node.items():
                if k in ["p", "t"] and isinstance(v, str):
                    raw_config["nodes"][i][k] = df[v].to_list()
                    logger.info(f"ノードの{k}データを読み込みました: {v}")

    # surfacesが存在する場合のみ処理
    if "surfaces" in raw_config:
        for i, surface in enumerate(raw_config["surfaces"]):
            if "solar" in surface and isinstance(surface["solar"], str):
                raw_config["surfaces"][i]["solar"] = df[surface["solar"]].to_list()
                logger.info(f"表面の日射量データを読み込みました: {surface['solar']}")

    # ノードの設定
    node_config = [{"key": "void"}]  # デフォルトノードを追加
    if "nodes" in raw_config:
        logger.info("ノードの解析を開始します")
        for node in raw_config["nodes"]:
            key = node["key"]
            comment = (
                key.split(COMMENT_DELIMITER, 1)[1].strip()
                if COMMENT_DELIMITER in key
                else ""
            )
            key = key.split(COMMENT_DELIMITER, 1)[0].strip()

            for sub_key in (
                [k.strip() for k in key.split(COMPOUND_DELIMITER)]
                if COMPOUND_DELIMITER in key
                else [key]
            ):
                if sub_key == "void":
                    raise ValueError("ノード名には'void'を使用できません。")
                node_dict = {**node, "key": sub_key, "comment": comment}
                node_config.append(node_dict)
                logger.info(f"ノードを解析しました: {sub_key}")
    else:
        raise ValueError("ノードの設定が見つかりませんでした。")

    # 計算フラグの設定
    for flag in ["p", "t", "x", "c"]:
        sim_config["calc_flag"][flag] = any(
            node.get(f"calc_{flag}", False) for node in node_config
        )

    # 換気ブランチの設定
    ventilation_config = []
    ventilation_branches_data = raw_config.get("ventilation_branches", [])
    if ventilation_branches_data:
        logger.info("換気ブランチの解析を開始します")
        
        # 入れ子構造の検証
        for i, branch in enumerate(ventilation_branches_data):
            if isinstance(branch, list):
                raise ValueError(f"ventilation_branches[{i}]に入れ子のリスト構造が検出されました。フラットなリスト構造にしてください。")
        
        for branch in ventilation_branches_data:
            key = branch["key"]
            comment = (
                key.split(COMMENT_DELIMITER, 1)[1].strip()
                if COMMENT_DELIMITER in key
                else ""
            )
            key = key.split(COMMENT_DELIMITER, 1)[0].strip()

            nodes = [n.strip() for n in key.split(CHAIN_DELIMITER)]
            branches = [
                f"{nodes[i]}{CHAIN_DELIMITER}{nodes[i+1]}"
                for i in range(len(nodes) - 1)
            ]

            for sub_key in branches:
                branch_dict = {**branch, "key": sub_key, "comment": comment}
                ventilation_config.append(branch_dict)
                logger.info(f"換気ブランチを解析しました: {sub_key}")
    else:
        logger.info("換気ブランチの設定が見つかりませんでした。")

    # 熱ブランチの設定
    thermal_config = []
    thermal_branches_data = raw_config.get("thermal_branches", [])
    if thermal_branches_data:
        logger.info("熱ブランチの解析を開始します")
        
        # 入れ子構造の検証
        for i, branch in enumerate(thermal_branches_data):
            if isinstance(branch, list):
                raise ValueError(f"thermal_branches[{i}]に入れ子のリスト構造が検出されました。フラットなリスト構造にしてください。")
        
        for branch in thermal_branches_data:
            key = branch["key"]
            comment = (
                key.split(COMMENT_DELIMITER, 1)[1].strip()
                if COMMENT_DELIMITER in key
                else ""
            )
            key = key.split(COMMENT_DELIMITER, 1)[0].strip()

            nodes = [n.strip() for n in key.split(CHAIN_DELIMITER)]
            branches = [
                f"{nodes[i] if nodes[i] != '' else 'void'}{CHAIN_DELIMITER}{nodes[i+1]}"
                for i in range(len(nodes) - 1)
            ]

            for sub_key in branches:
                branch_dict = {**branch, "key": sub_key, "comment": comment}
                thermal_config.append(branch_dict)
                logger.info(f"熱ブランチを解析しました: {sub_key}")
    else:
        logger.info("熱ブランチの設定が見つかりませんでした。")

    # 表面の設定（キー内コメント'||'の除去に対応）
    surface_config = []
    raw_surface_config = raw_config.get("surfaces", [])
    if raw_surface_config:
        logger.info("表面の解析を開始します")
        for surface in raw_surface_config:
            key = surface["key"]
            comment = (
                key.split(COMMENT_DELIMITER, 1)[1].strip()
                if COMMENT_DELIMITER in key
                else ""
            )
            key = key.split(COMMENT_DELIMITER, 1)[0].strip()
            surface_config.append({**surface, "key": key, "comment": comment})
    else:
        logger.info("表面の設定が見つかりませんでした。")

    # 空調の設定
    aircon_config = raw_config.get("aircon", [])
    if aircon_config:
        logger.info("エアコンの解析を開始します")
    else:
        logger.info("エアコンの設定が見つかりませんでした。")

    # 設定ファイルの保存
    output_json = {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
        "surfaces": surface_config,
        "aircon": aircon_config,
    }

    # 設定ファイルの保存
    output_json = {
        "simulation": sim_config,
        "nodes": node_config,
        "ventilation_branches": ventilation_config,
        "thermal_branches": thermal_config,
        "surfaces": surface_config,
        "aircon": aircon_config,
    }

    logger.info("設定データの処理が完了しました")
    return output_json
  
  except Exception as e:
      logger.exception("エラーが発生しました: %s", e)
      raise
