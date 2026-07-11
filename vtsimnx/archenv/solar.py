"""日射関連の公開ファサード（太陽位置 + 日射熱取得）。"""
from .solar_position import sun_loc, astro_sun_loc
from .solar_gain import solar_gain_by_angles, solar_gain_by_angles_with_shade

__all__ = [
    "sun_loc",
    "astro_sun_loc",
    "solar_gain_by_angles",
    "solar_gain_by_angles_with_shade",
]
