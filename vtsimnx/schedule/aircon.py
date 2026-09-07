"""
暖冷房（aircon）関連のスケジュール。

- 暖冷房期間(period_1..8)
- 休日フラグ(holiday)
- 8760生成関数(make_8760_data / make_8760_by_holiday)
- 空調モード(ac_mode) / 設定温度(pre_tmp) / 設定湿度(pre_rh)

プロファイル種別:
- 個別間欠（LD / 寝室 / 子供室1 / 子供室2）: 省エネ基準系の在室間欠運転
- 全館空調: 暖冷房期間中は平日・休日とも 24 時間連続運転（中間期は停止）
"""

from __future__ import annotations

from typing import Any, Dict

from .common import (
    HOURS_PER_DAY,
    holiday,
    make_8760_by_holiday,
    make_8760_data,
    period_1,
    period_2,
    period_3,
    period_4,
    period_5,
    period_6,
    period_7,
    period_8,
)

# 空調モード（本モジュールの ac_mode の値）
# 0: 停止 / 1: 暖房 / 2: 冷房 / 3:自動
AC_MODE_STOP = 0
AC_MODE_HEATING = 1
AC_MODE_COOLING = 2
AC_MODE_AUTO = 3

# 全館連続運転用のキー（DUCT_CENTRAL 等）
WHOLE_HOUSE_KEY = "全館空調"

ac_mode_profiles: Dict[str, Any] = {
    "LD": {
        "暖房": {"平日": [], "休日": []},
        "冷房": {"平日": [], "休日": []},
    },
    "寝室": {
        "暖房": {"平日": [], "休日": []},
        "冷房": {"平日": [], "休日": []},
    },
    "子供室1": {
        "暖房": {"平日": [], "休日": []},
        "冷房": {"平日": [], "休日": []},
    },
    "子供室2": {
        "暖房": {"平日": [], "休日": []},
        "冷房": {"平日": [], "休日": []},
    },
    WHOLE_HOUSE_KEY: {
        "暖房": {"平日": [], "休日": []},
        "冷房": {"平日": [], "休日": []},
    },
}

# --- 個別間欠空調 ---

# 主たる居室の暖冷房スケジュール
ac_mode_profiles["LD"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
ac_mode_profiles["LD"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0]
ac_mode_profiles["LD"]["冷房"]["平日"] = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
ac_mode_profiles["LD"]["冷房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0]


# 寝室の暖冷房スケジュール
ac_mode_profiles["寝室"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ac_mode_profiles["寝室"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
ac_mode_profiles["寝室"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
ac_mode_profiles["寝室"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]


# 子供室1の暖冷房スケジュール
ac_mode_profiles["子供室1"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
ac_mode_profiles["子供室1"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0]
ac_mode_profiles["子供室1"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2]
ac_mode_profiles["子供室1"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2]


# 子供室2の暖冷房スケジュール
ac_mode_profiles["子供室2"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0]
ac_mode_profiles["子供室2"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]
ac_mode_profiles["子供室2"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2]
ac_mode_profiles["子供室2"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]


# --- 全館空調（暖冷房期間中 24 時間連続） ---
# 中間期は make_8760_data の default(=STOP) になる。
ac_mode_profiles[WHOLE_HOUSE_KEY]["暖房"]["平日"] = [AC_MODE_HEATING] * HOURS_PER_DAY
ac_mode_profiles[WHOLE_HOUSE_KEY]["暖房"]["休日"] = [AC_MODE_HEATING] * HOURS_PER_DAY
ac_mode_profiles[WHOLE_HOUSE_KEY]["冷房"]["平日"] = [AC_MODE_COOLING] * HOURS_PER_DAY
ac_mode_profiles[WHOLE_HOUSE_KEY]["冷房"]["休日"] = [AC_MODE_COOLING] * HOURS_PER_DAY


pre_tmp_profiles: Dict[str, Any] = {
    "LD": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "寝室": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "子供室1": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "子供室2": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    WHOLE_HOUSE_KEY: {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
}

# 相対湿度の設定値[%]
# 既定は全時間 60% とする（必要に応じて room/season/day_type ごとに差し替え可能）
rh_profiles = {
    "LD": {"暖房": {"平日": [60.0] * 24, "休日": [60.0] * 24}, "冷房": {"平日": [60.0] * 24, "休日": [60.0] * 24}},
    "寝室": {"暖房": {"平日": [60.0] * 24, "休日": [60.0] * 24}, "冷房": {"平日": [60.0] * 24, "休日": [60.0] * 24}},
    "子供室1": {"暖房": {"平日": [60.0] * 24, "休日": [60.0] * 24}, "冷房": {"平日": [60.0] * 24, "休日": [60.0] * 24}},
    "子供室2": {"暖房": {"平日": [60.0] * 24, "休日": [60.0] * 24}, "冷房": {"平日": [60.0] * 24, "休日": [60.0] * 24}},
    WHOLE_HOUSE_KEY: {
        "暖房": {"平日": [60.0] * 24, "休日": [60.0] * 24},
        "冷房": {"平日": [60.0] * 24, "休日": [60.0] * 24},
    },
}

# 主たる居室の設定温度
pre_tmp_profiles["LD"]["暖房"]["平日"] = [20.0] * 24
pre_tmp_profiles["LD"]["暖房"]["休日"] = [20.0] * 24
pre_tmp_profiles["LD"]["冷房"]["平日"] = [27.0] * 24
pre_tmp_profiles["LD"]["冷房"]["休日"] = [27.0] * 24


# 寝室の設定温度
pre_tmp_profiles["寝室"]["暖房"]["平日"] = [20.0] * 24
pre_tmp_profiles["寝室"]["暖房"]["休日"] = [20.0] * 24
pre_tmp_profiles["寝室"]["冷房"]["平日"] = [28.0] * 24
pre_tmp_profiles["寝室"]["冷房"]["休日"] = [28.0] * 24


# 子供室1の設定温度
pre_tmp_profiles["子供室1"]["暖房"]["平日"] = [20.0] * 24
pre_tmp_profiles["子供室1"]["暖房"]["休日"] = [20.0] * 24
pre_tmp_profiles["子供室1"]["冷房"]["平日"] = [28.0] * 8 + [27.0] * 16
pre_tmp_profiles["子供室1"]["冷房"]["休日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1


# 子供室2の設定温度
pre_tmp_profiles["子供室2"]["暖房"]["平日"] = [20] * 24
pre_tmp_profiles["子供室2"]["暖房"]["休日"] = [20] * 24
pre_tmp_profiles["子供室2"]["冷房"]["平日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1
pre_tmp_profiles["子供室2"]["冷房"]["休日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1


# 全館空調の設定温度（代表室 LD 相当: 暖房 20℃ / 冷房 27℃）
pre_tmp_profiles[WHOLE_HOUSE_KEY]["暖房"]["平日"] = [20.0] * HOURS_PER_DAY
pre_tmp_profiles[WHOLE_HOUSE_KEY]["暖房"]["休日"] = [20.0] * HOURS_PER_DAY
pre_tmp_profiles[WHOLE_HOUSE_KEY]["冷房"]["平日"] = [27.0] * HOURS_PER_DAY
pre_tmp_profiles[WHOLE_HOUSE_KEY]["冷房"]["休日"] = [27.0] * HOURS_PER_DAY


_REGIONS = {
    "region1": period_1,
    "region2": period_2,
    "region3": period_3,
    "region4": period_4,
    "region5": period_5,
    "region6": period_6,
    "region7": period_7,
    "region8": period_8,
}


def _profile_tuple(profiles: Dict[str, Any], key: str, default):
    p = profiles[key]
    return (
        p["暖房"]["平日"],
        p["暖房"]["休日"],
        p["冷房"]["平日"],
        p["冷房"]["休日"],
        default,
    )


_AC_MODE_PROFILES = {
    key: _profile_tuple(ac_mode_profiles, key, AC_MODE_STOP)
    for key in ac_mode_profiles
}


_PRE_TMP_PROFILES = {
    key: _profile_tuple(pre_tmp_profiles, key, 20.0)
    for key in pre_tmp_profiles
}


_PRE_RH_PROFILES = {
    key: _profile_tuple(rh_profiles, key, 60.0)
    for key in rh_profiles
}


def _validate_aircon_profiles():
    expected_modes = {AC_MODE_STOP, AC_MODE_HEATING, AC_MODE_COOLING, AC_MODE_AUTO}

    for room, by_season in ac_mode_profiles.items():
        for season, by_day in by_season.items():
            for day_type in ("平日", "休日"):
                prof = by_day[day_type]
                if len(prof) != HOURS_PER_DAY:
                    raise ValueError(
                        f"ac_mode_profiles[{room}][{season}][{day_type}] length must be {HOURS_PER_DAY}, got {len(prof)}"
                    )
                invalid = [v for v in prof if v not in expected_modes]
                if invalid:
                    raise ValueError(
                        f"ac_mode_profiles[{room}][{season}][{day_type}] contains invalid mode values: {sorted(set(invalid))}"
                    )

    for room, by_season in pre_tmp_profiles.items():
        for season, by_day in by_season.items():
            for day_type in ("平日", "休日"):
                prof = by_day[day_type]
                if len(prof) != HOURS_PER_DAY:
                    raise ValueError(
                        f"pre_tmp_profiles[{room}][{season}][{day_type}] length must be {HOURS_PER_DAY}, got {len(prof)}"
                    )

    for room, by_season in rh_profiles.items():
        for season, by_day in by_season.items():
            for day_type in ("平日", "休日"):
                prof = by_day[day_type]
                if len(prof) != HOURS_PER_DAY:
                    raise ValueError(
                        f"rh_profiles[{room}][{season}][{day_type}] length must be {HOURS_PER_DAY}, got {len(prof)}"
                    )


def build_ac_mode(*, holiday_days=holiday):
    """
    地域×部屋の空調モード（8760）を生成する。
    戻り値の構造は従来の ac_mode と同一。
    """
    out = {}
    for region, period in _REGIONS.items():
        out[region] = {
            room: make_8760_data(period, holiday_days, w_h, h_h, w_c, h_c, default)
            for room, (w_h, h_h, w_c, h_c, default) in _AC_MODE_PROFILES.items()
        }
    return out


def build_pre_tmp(*, holiday_days=holiday):
    """
    地域×部屋の設定温度（8760）を生成する。
    戻り値の構造は従来の pre_tmp と同一。
    """
    out = {}
    for region, period in _REGIONS.items():
        out[region] = {
            room: make_8760_data(period, holiday_days, w_h, h_h, w_c, h_c, default)
            for room, (w_h, h_h, w_c, h_c, default) in _PRE_TMP_PROFILES.items()
        }
    return out


def build_pre_rh(*, holiday_days=holiday):
    """
    地域×部屋の設定湿度（8760）を生成する。
    戻り値の構造は pre_tmp / ac_mode と同一。
    """
    out = {}
    for region, period in _REGIONS.items():
        out[region] = {
            room: make_8760_data(period, holiday_days, w_h, h_h, w_c, h_c, default)
            for room, (w_h, h_h, w_c, h_c, default) in _PRE_RH_PROFILES.items()
        }
    return out


_validate_aircon_profiles()
ac_mode = build_ac_mode()
pre_tmp = build_pre_tmp()
pre_rh = build_pre_rh()


__all__ = [
    # constants
    "AC_MODE_HEATING",
    "AC_MODE_COOLING",
    "AC_MODE_STOP",
    "AC_MODE_AUTO",
    "WHOLE_HOUSE_KEY",
    # daily profiles
    "ac_mode_profiles",
    "pre_tmp_profiles",
    "rh_profiles",
    # profiles
    "ac_mode",
    "pre_tmp",
    "pre_rh",
    # functions / outputs
    "make_8760_data",
    "make_8760_by_holiday",
    "build_ac_mode",
    "build_pre_tmp",
    "build_pre_rh",
]


