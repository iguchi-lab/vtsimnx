from __future__ import annotations

# ------------------------------
# 表面の種類の対応関係 と 物性定数
# ------------------------------
SURFACE_PAIR = {
    "wall": "wall",
    "floor": "ceiling",
    "ceiling": "floor",
    "glass": "glass",
}
SURFACE_PART_ALIASES = {
    "window": "glass",
}
SOLAR_TARGET_PARTS = frozenset(["wall", "floor", "ceiling"])
NOCTURNAL_TARGET_PARTS = frozenset(["wall", "floor", "ceiling", "glass"])

DEFAULT_ALPHA_I = 4.4   # 室内側表面の対流熱伝達率
DEFAULT_ALPHA_O = 20.3  # 室外側表面の対流熱伝達率
DEFAULT_ALPHA_R = 4.7   # 室内表面間の放射熱伝達率 [W/m2/K]（両面の長波放射率0.9は既に含む。0.9/0.8は掛けない）
DEFAULT_ETA_SW = 0.8   # 短波（日射）の吸収率（外壁日射・ガラス透過日射の床・壁への吸収）
DEFAULT_ETA_LW = 0.9    # 長波の吸収率（夜間放射・発熱の放射配分で表面が吸収するとき。室内表面間の4.7には不要）
DEFAULT_EPSILON_LW = 0.9  # 長波放射率（夜間放射の放出側。室内表面間の4.7には既に含まれる）
# 空気の体積熱容量 [J/(m³·K)]。中空層・通気層の標準値（1298 J/(m³·K)。SimHeat で採用されている値）。
DEFAULT_AIR_V_CAPA = 1298.0
