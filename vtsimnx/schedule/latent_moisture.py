"""
潜熱由来の水分発生（主に人体）のスケジュール（単位: kg/h）。

現状は「人体」のみを対象とし、時間プロファイルは顕熱スケジュールと同一、
強度だけ「56 W/人」を蒸発潜熱で割って kg/h に換算している。
"""

import numpy as np

from vtsimnx.archenv import Vap_L  # [kJ/kg]

from .common import HOURS_PER_DAY, holiday, make_8760_by_holiday
from .sensible_heat import sensible_heat_profiles


# 1 人あたりの潜熱発熱 [W]
LATENT_POWER_PER_PERSON_W = 56.0

# 蒸発潜熱 [J/kg]
LATENT_HEAT_J_PER_KG = Vap_L * 1000.0

# 1 人あたりの水分発生量 [kg/h]
LATENT_KG_PER_H_PER_PERSON = LATENT_POWER_PER_PERSON_W * 3600.0 / LATENT_HEAT_J_PER_KG


def _extract_people_profiles():
    """
    顕熱スケジュールから「人体」のプロファイル（人数スケール）だけを抽出し、
    人数×[kg/h/人] で水分発生量に換算した 24h プロファイルを作る。
    """
    latent_profiles: dict[str, dict[str, dict[str, list[float]]]] = {}

    for room, by_use in sensible_heat_profiles.items():
        people = by_use.get("人体")
        if people is None:
            continue

        weekday = np.asarray(people["平日"], dtype="float64")
        holiday_ = np.asarray(people["休日"], dtype="float64")

        # 顕熱側の人体プロファイルは
        #   （人数プロファイル）× 63[W/人]
        # になっているため、63 で割って人数に戻す。
        people_count_weekday = weekday / 63.0
        people_count_holiday = holiday_ / 63.0

        latent_weekday = (people_count_weekday * LATENT_KG_PER_H_PER_PERSON).tolist()
        latent_holiday = (people_count_holiday * LATENT_KG_PER_H_PER_PERSON).tolist()

        if len(latent_weekday) != HOURS_PER_DAY or len(latent_holiday) != HOURS_PER_DAY:
            raise ValueError(f"latent_moisture: unexpected profile length for room {room!r}")

        latent_profiles.setdefault(room, {})["人体"] = {
            "平日": latent_weekday,
            "休日": latent_holiday,
        }

    return latent_profiles


latent_moisture_profiles = _extract_people_profiles()


# 台所機器からの水蒸気発生（最大 50 g/h = 0.05 kg/h）
_KITCHEN_APPLIANCE_MAX_KG_PER_H = 0.05

_kitchen_weekday_pct = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    50.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    100.00, 0.00, 0.00, 0.00, 0.00, 0.00,
], dtype="float64")

_kitchen_holiday_pct = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 50.00, 0.00, 0.00, 0.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 100.00,
    0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
], dtype="float64")

_kitchen_weekday_kgph = (_kitchen_weekday_pct / 100.0 * _KITCHEN_APPLIANCE_MAX_KG_PER_H).tolist()
_kitchen_holiday_kgph = (_kitchen_holiday_pct / 100.0 * _KITCHEN_APPLIANCE_MAX_KG_PER_H).tolist()

latent_moisture_profiles.setdefault("台所", {})["機器"] = {
    "平日": _kitchen_weekday_kgph,
    "休日": _kitchen_holiday_kgph,
}


def _build_room_latent_moisture(by_use, *, holiday_days):
    return {
        use_name: make_8760_by_holiday(
            holiday_days,
            profs["平日"],
            profs["休日"],
            default=0.0,
        )
        for use_name, profs in by_use.items()
    }


def _validate_latent_moisture_profiles(profiles):
    for room, by_use in profiles.items():
        for use_name, profs in by_use.items():
            if "平日" not in profs or "休日" not in profs:
                raise ValueError(
                    f"latent_moisture_profiles[{room}][{use_name}] は '平日' と '休日' を持つ必要があります。"
                )
            if len(profs["平日"]) != HOURS_PER_DAY:
                raise ValueError(
                    f"latent_moisture_profiles[{room}][{use_name}]['平日'] length must be {HOURS_PER_DAY}, "
                    f"got {len(profs['平日'])}"
                )
            if len(profs["休日"]) != HOURS_PER_DAY:
                raise ValueError(
                    f"latent_moisture_profiles[{room}][{use_name}]['休日'] length must be {HOURS_PER_DAY}, "
                    f"got {len(profs['休日'])}"
                )


def build_latent_moisture_schedule(*, holiday_days=holiday):
    """
    潜熱由来の水分発生スケジュール（kg/h）を生成する。

    戻り値の構造:
      {
        "LD": {
          "人体": np.ndarray(shape=(8760,)),  # kg/h
          ...
        },
        ...
      }
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for room, by_use in latent_moisture_profiles.items():
        out[room] = _build_room_latent_moisture(by_use, holiday_days=holiday_days)
    return out


_validate_latent_moisture_profiles(latent_moisture_profiles)
latent_moisture = build_latent_moisture_schedule()


__all__ = [
    "latent_moisture_profiles",
    "build_latent_moisture_schedule",
    "latent_moisture",
]

