"""
schedule 共通部品。

- 休日フラグ(holiday)
- 暖冷房期間(period_1..8)
- 8760生成関数(make_8760_data / make_8760_by_holiday)
"""

# 暖冷房期間（365要素 / 1:暖房期, 0:非空調期, -1:冷房期）
# ※ここは「季節区分」です。空調の出力モード(1/2/3)とは別物です。
period_1 = [1] * 158 + [0] * 32 + [-1] * 53 + [0] * 23 + [1] * 99
period_2 = [1] * 155 + [0] * 40 + [-1] * 48 + [0] * 25 + [1] * 97
period_3 = [1] * 151 + [0] * 39 + [-1] * 53 + [0] * 29 + [1] * 93
period_4 = [1] * 150 + [0] * 40 + [-1] * 53 + [0] * 30 + [1] * 92
period_5 = [1] * 135 + [0] * 51 + [-1] * 57 + [0] * 39 + [1] * 83
period_6 = [1] * 111 + [0] * 38 + [-1] * 117 + [0] * 41 + [1] * 58
period_7 = [1] * 86 + [0] * 48 + [-1] * 152 + [0] * 43 + [1] * 36
period_8 = [1] * 0 + [0] * 83 + [-1] * 265 + [0] * 17 + [1] * 0


# 休日（365要素）
holiday = [
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
    1,
]


def make_8760_data(ac_mode, holiday, w_h, h_h, w_c, h_c, default):
    """
    365日×24時間=8760 の時系列データを生成する。

    注意: 第1引数の ac_mode は「季節区分（暖冷房期間）」です。
      -  1: 暖房期
      -  0: 非空調期（default を24時間分）
      - -1: 冷房期

    本プロジェクトでは、w_*/h_*（出力値の24hプロファイル）は
    1:暖房 / 2:冷房 / 3:停止 のコードを使っています（必要なら別コードでも可）。

    holiday の値の意味:
      - 1: 休日
      - 0: 平日

    入力の w_*/h_* は「1日24要素」のプロファイル。
    """
    assert len(ac_mode) == 365, f"ac_mode length must be 365, got {len(ac_mode)}"
    assert len(holiday) == 365, f"holiday length must be 365, got {len(holiday)}"
    for name, seq in (("w_h", w_h), ("h_h", h_h), ("w_c", w_c), ("h_c", h_c)):
        assert len(seq) == 24, f"{name} length must be 24, got {len(seq)}"

    default_day = [default] * 24
    profile_by_key = {
        (1, 0): w_h,  # 暖房・平日
        (1, 1): h_h,  # 暖房・休日
        (-1, 0): w_c,  # 冷房・平日
        (-1, 1): h_c,  # 冷房・休日
    }

    output = []
    for day in range(365):
        mode = ac_mode[day]
        if mode == 0:
            output.extend(default_day)
            continue

        is_holiday = 1 if holiday[day] else 0
        prof = profile_by_key.get((mode, is_holiday))
        assert prof is not None, f"unexpected ac_mode/holiday combination: {(mode, is_holiday)}"
        output.extend(prof)

    return output


def make_8760_by_holiday(holiday, weekday_24, holiday_24=None, default=0.0):
    """
    365日×24時間=8760 の時系列データを生成する（休日/平日だけで切り替える簡易版）。

    - holiday[i] が truthy なら holiday_24 を採用（Noneなら weekday_24 を使う）
    - holiday[i] が falsy なら weekday_24 を採用
    - weekday_24 / holiday_24 は「1日24要素」のプロファイル（値の意味は呼び出し側に依存）
    """
    assert len(holiday) == 365, f"holiday length must be 365, got {len(holiday)}"
    assert len(weekday_24) == 24, f"weekday_24 length must be 24, got {len(weekday_24)}"
    if holiday_24 is not None:
        assert len(holiday_24) == 24, f"holiday_24 length must be 24, got {len(holiday_24)}"

    out = []
    for day in range(365):
        if holiday[day]:
            out.extend(holiday_24 if holiday_24 is not None else weekday_24)
        else:
            out.extend(weekday_24 if weekday_24 is not None else [default] * 24)
    return out


__all__ = [
    # calendar
    "period_1",
    "period_2",
    "period_3",
    "period_4",
    "period_5",
    "period_6",
    "period_7",
    "period_8",
    "holiday",
    # helpers
    "make_8760_data",
    "make_8760_by_holiday",
]


