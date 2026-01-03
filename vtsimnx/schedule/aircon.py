"""
暖冷房（aircon）関連のスケジュール。

- 暖冷房期間(period_1..8)
- 休日フラグ(holiday)
- 8760生成関数(make_8760_data / make_8760_by_holiday)
- 空調モード(ac_mode) / 設定温度(pre_tmp)
"""

from .common import (
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

ac_mode_profiles = {
    "LD": {
        "暖房": {"平日": [], "休日": []}, 
        "冷房": {"平日": [], "休日": []}
    },
    "寝室": {
        "暖房": {"平日": [], "休日": []}, 
        "冷房": {"平日": [], "休日": []}
    },
    "子供室1": {
        "暖房": {"平日": [], "休日": []}, 
        "冷房": {"平日": [], "休日": []}
    },
    "子供室2": {
        "暖房": {"平日": [], "休日": []}, 
        "冷房": {"平日": [], "休日": []}
    }
}

# 主たる居室の暖冷房スケジュール
ac_mode_profiles["LD"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
ac_mode_profiles["LD"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 3]
ac_mode_profiles["LD"]["冷房"]["平日"] = [0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]
ac_mode_profiles["LD"]["冷房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 3]


# 寝室の暖冷房スケジュール
ac_mode_profiles["寝室"]["暖房"]["平日"] = [0, 3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3]
ac_mode_profiles["寝室"]["暖房"]["休日"] = [0, 3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3 ,3]
ac_mode_profiles["寝室"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
ac_mode_profiles["寝室"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]


# 子供室1の暖冷房スケジュール
ac_mode_profiles["子供室1"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]
ac_mode_profiles["子供室1"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 3]
ac_mode_profiles["子供室1"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2]
ac_mode_profiles["子供室1"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 2, 2, 2, 2]


# 子供室2の暖冷房スケジュール
ac_mode_profiles["子供室2"]["暖房"]["平日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 3]
ac_mode_profiles["子供室2"]["暖房"]["休日"] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3]
ac_mode_profiles["子供室2"]["冷房"]["平日"] = [2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2]
ac_mode_profiles["子供室2"]["冷房"]["休日"] = [2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]


pre_tmp_profiles = {
    "LD": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "寝室": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "子供室1": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
    "子供室2": {"暖房": {"平日": [], "休日": []}, "冷房": {"平日": [], "休日": []}},
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
pre_tmp_profiles["子供室1"]["冷房"]["平日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1
pre_tmp_profiles["子供室1"]["冷房"]["休日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1


# 子供室2の設定温度
pre_tmp_profiles["子供室2"]["暖房"]["平日"] = [20] * 24
pre_tmp_profiles["子供室2"]["暖房"]["休日"] = [20] * 24
pre_tmp_profiles["子供室2"]["冷房"]["平日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1
pre_tmp_profiles["子供室2"]["冷房"]["休日"] = [28.0] * 8 + [27.0] * 15 + [28.0] * 1


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


_AC_MODE_PROFILES = {
    "LD": (
        ac_mode_profiles["LD"]["暖房"]["平日"],
        ac_mode_profiles["LD"]["暖房"]["休日"],
        ac_mode_profiles["LD"]["冷房"]["平日"],
        ac_mode_profiles["LD"]["冷房"]["休日"],
        AC_MODE_STOP,
    ),
    "寝室": (
        ac_mode_profiles["寝室"]["暖房"]["平日"],
        ac_mode_profiles["寝室"]["暖房"]["休日"],
        ac_mode_profiles["寝室"]["冷房"]["平日"],
        ac_mode_profiles["寝室"]["冷房"]["休日"],
        AC_MODE_STOP,
    ),
    "子供室1": (
        ac_mode_profiles["子供室1"]["暖房"]["平日"],
        ac_mode_profiles["子供室1"]["暖房"]["休日"],
        ac_mode_profiles["子供室1"]["冷房"]["平日"],
        ac_mode_profiles["子供室1"]["冷房"]["休日"],
        AC_MODE_STOP,
    ),
    "子供室2": (
        ac_mode_profiles["子供室2"]["暖房"]["平日"],
        ac_mode_profiles["子供室2"]["暖房"]["休日"],
        ac_mode_profiles["子供室2"]["冷房"]["平日"],
        ac_mode_profiles["子供室2"]["冷房"]["休日"],
        AC_MODE_STOP,
    ),
}


_PRE_TMP_PROFILES = {
    "LD": (pre_tmp_profiles["LD"]["暖房"]["平日"], pre_tmp_profiles["LD"]["暖房"]["休日"], pre_tmp_profiles["LD"]["冷房"]["平日"], pre_tmp_profiles["LD"]["冷房"]["休日"], 20.0),
    "寝室": (pre_tmp_profiles["寝室"]["暖房"]["平日"], pre_tmp_profiles["寝室"]["暖房"]["休日"], pre_tmp_profiles["寝室"]["冷房"]["平日"], pre_tmp_profiles["寝室"]["冷房"]["休日"], 20.0),
    "子供室1": (pre_tmp_profiles["子供室1"]["暖房"]["平日"], pre_tmp_profiles["子供室1"]["暖房"]["休日"], pre_tmp_profiles["子供室1"]["冷房"]["平日"], pre_tmp_profiles["子供室1"]["冷房"]["休日"], 20.0),
    "子供室2": (pre_tmp_profiles["子供室2"]["暖房"]["平日"], pre_tmp_profiles["子供室2"]["暖房"]["休日"], pre_tmp_profiles["子供室2"]["冷房"]["平日"], pre_tmp_profiles["子供室2"]["冷房"]["休日"], 20.0),
}


def build_ac_mode(*, holiday_days=holiday):
    """
    地域×部屋の空調モード（8760）を生成する。
    戻り値の構造は従来の ac_mode と同一。
    """
    out = {}
    for region, period in _REGIONS.items():
        rooms = {}
        for room, (w_h, h_h, w_c, h_c, default) in _AC_MODE_PROFILES.items():
            rooms[room] = make_8760_data(period, holiday_days, w_h, h_h, w_c, h_c, default)
        out[region] = rooms
    return out


def build_pre_tmp(*, holiday_days=holiday):
    """
    地域×部屋の設定温度（8760）を生成する。
    戻り値の構造は従来の pre_tmp と同一。
    """
    out = {}
    for region, period in _REGIONS.items():
        rooms = {}
        for room, (w_h, h_h, w_c, h_c, default) in _PRE_TMP_PROFILES.items():
            rooms[room] = make_8760_data(period, holiday_days, w_h, h_h, w_c, h_c, default)
        out[region] = rooms
    return out


ac_mode = build_ac_mode()
pre_tmp = build_pre_tmp()


__all__ = [
    # constants
    "AC_MODE_HEATING",
    "AC_MODE_COOLING",
    "AC_MODE_STOP",
    # profiles
    "ac_mode",
    "pre_tmp",
    # functions / outputs
    "make_8760_data",
    "make_8760_by_holiday",
    "build_ac_mode",
    "build_pre_tmp",
    "ac_mode",
    "pre_tmp",
]


