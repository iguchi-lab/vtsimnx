"""換気・内部発熱・空調の選択部品。取り出したリストは呼び出し側で編集できる。"""

from __future__ import annotations

from typing import Any, Mapping

from .envelope import ROOM_VOLUME

HEAT_RATIO: dict[str, dict[str, float]] = {
    "人体": {"convection": 0.50, "radiation": 0.50},
    "蛍光灯": {"convection": 0.60, "radiation": 0.40},
    "白熱灯": {"convection": 0.40, "radiation": 0.60},
    "機器": {"convection": 0.60, "radiation": 0.40},
}


def make_aircon_specs() -> dict[str, dict[str, Any]]:
    return {
        "highspec": {
            "Q": {
                "cooling": {"min": 0.700, "rtd": 2.200, "max": 3.300},
                "heating": {"min": 0.700, "rtd": 2.500, "max": 5.400},
            },
            "P": {
                "cooling": {"min": 0.095, "rtd": 0.395, "max": 0.780},
                "heating": {"min": 0.095, "rtd": 0.390, "max": 1.360},
            },
            "V_inner": {"cooling": {"rtd": 12.1 / 60}, "heating": {"rtd": 13.1 / 60}},
            "V_outer": {"cooling": {"rtd": 28.2 / 60}, "heating": {"rtd": 25.5 / 60}},
        },
        "standard": {
            "Q": {
                "cooling": {"min": 0.900, "rtd": 2.200, "max": 2.800},
                "heating": {"min": 0.900, "rtd": 2.200, "max": 3.600},
            },
            "P": {
                "cooling": {"min": 0.170, "rtd": 0.455, "max": 0.745},
                "heating": {"min": 0.135, "rtd": 0.385, "max": 1.070},
            },
            "V_inner": {"cooling": {"rtd": 12.0 / 60}, "heating": {"rtd": 12.0 / 60}},
            "V_outer": {"cooling": {"rtd": 27.6 / 60}, "heating": {"rtd": 22.5 / 60}},
        },
        "rac": {
            "Q": {"cooling": {"rtd": 2.2, "max": 2.8}, "heating": {"rtd": 2.2, "max": 3.6}},
            "P": {"cooling": {"rtd": 0.455}, "heating": {"rtd": 0.385}},
            "dualcompressor": False,
        },
        "rac_standard": {
            "Q": {
                "cooling": {"rtd": 5.60000, "max": 5.94462},
                "heating": {"rtd": 6.68530, "max": 10.04705},
            },
            "P": {
                "cooling": {"rtd": 5.60000 / 3.2432},
                "heating": {"rtd": 6.68530 / 4.1573},
            },
            "dualcompressor": False,
        },
    }


def make_ventilation_24h(
    room_volume: Mapping[str, float] | None = None,
    vol_schedule: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """24時間換気・局所排気・床下小屋裏換気。隙間換気は含まない。"""
    if room_volume is None:
        room_volume = ROOM_VOLUME
    if vol_schedule is None:
        from vtsimnx.schedule import vol as default_vol
        schedule = default_vol
    else:
        schedule = vol_schedule
    return [
        {"key": "外部->和室->ホール", "vol": 20 / 3600},
        {"key": "外部->台所->LD->ホール", "vol": 20 / 3600},
        {"key": "外部->LD->ホール", "vol": 40 / 3600},
        {"key": "ホール->外部", "vol": 80 / 3 / 3600},
        {"key": "ホール->洗面所->外部", "vol": 80 / 3 / 3600},
        {"key": "ホール->1階トイレ->外部", "vol": 80 / 3 / 3600},
        {"key": "外部->クローゼット->寝室->2階ホール", "vol": 20 / 3600},
        {"key": "外部->寝室->2階ホール", "vol": 20 / 3600},
        {"key": "外部->子供室1->2階ホール", "vol": 20 / 3600},
        {"key": "外部->子供室2->2階ホール", "vol": 20 / 3600},
        {"key": "2階ホール->2階トイレ->外部", "vol": 40 / 3600},
        {"key": "2階ホール->外部", "vol": 40 / 3600},
        {"key": "ホール->2階ホール->ホール", "vol": 273 / 3600},
        {"key": "LD->台所->LD", "vol": 5000 / 3600},
        {"key": "外部->台所->外部", "vol": schedule["LD"]},
        {"key": "外部->ホール->洗面所->浴室->外部", "vol": schedule["浴室"]},
        {"key": "外部->ホール->1階トイレ->外部", "vol": schedule["1階トイレ"]},
        {"key": "外部->床下->外部", "vol": room_volume["床下"] * 5.0 / 3600},
        {"key": "外部->切妻->外部", "vol": room_volume["切妻"] * 5.0 / 3600},
        {"key": "外部->片流れ(3)->外部", "vol": room_volume["片流れ(3)"] * 5.0 / 3600},
        {"key": "外部->片流れ(4)->外部", "vol": room_volume["片流れ(4)"] * 5.0 / 3600},
    ]


def make_aircon_ac4(region: str = "region6") -> list[dict[str, Any]]:
    """LD・寝室・子供室1・子供室2に各1台。仕様や台数は取り出したあと差し替えてよい。"""
    from vtsimnx.schedule import ac_mode, pre_tmp

    specs = make_aircon_specs()
    return [
        {
            "key": "LDエアコン",
            "set": "LD",
            "in": "LD",
            "out": "LD",
            "outside": "外部",
            "mode": ac_mode[region]["LD"],
            "pre_temp": pre_tmp[region]["LD"],
            "model": "RAC",
            "ac_spec": specs["rac_standard"],
        },
        {
            "key": "寝室エアコン",
            "set": "寝室",
            "in": "寝室",
            "out": "寝室",
            "outside": "外部",
            "mode": ac_mode[region]["寝室"],
            "pre_temp": pre_tmp[region]["寝室"],
            "model": "CRIEPI",
            "ac_spec": specs["highspec"],
        },
        {
            "key": "子供室1エアコン",
            "set": "子供室1",
            "in": "子供室1",
            "out": "子供室1",
            "outside": "外部",
            "mode": ac_mode[region]["子供室1"],
            "pre_temp": pre_tmp[region]["子供室1"],
            "model": "RAC",
            "ac_spec": specs["rac"],
        },
        {
            "key": "子供室2エアコン",
            "set": "子供室2",
            "in": "子供室2",
            "out": "子供室2",
            "outside": "外部",
            "mode": ac_mode[region]["子供室2"],
            "pre_temp": pre_tmp[region]["子供室2"],
            "model": "CRIEPI",
            "ac_spec": specs["highspec"],
        },
    ]


def make_heat_sources() -> list[dict[str, Any]]:
    from vtsimnx.schedule import sensible_heat as s_heat

    ratio = HEAT_RATIO
    return [
        {"key": "LD_人体", "room": "LD", "generation_rate": s_heat["LD"]["人体"], **ratio["人体"]},
        {"key": "LD_照明", "room": "LD", "generation_rate": s_heat["LD"]["照明"], **ratio["蛍光灯"]},
        {"key": "LD_機器", "room": "LD", "generation_rate": s_heat["LD"]["機器"], **ratio["機器"]},
        {"key": "台所_照明", "room": "台所", "generation_rate": s_heat["台所"]["照明"], **ratio["蛍光灯"]},
        {"key": "台所_機器1", "room": "台所", "generation_rate": s_heat["台所"]["機器1"], **ratio["機器"]},
        {"key": "台所_機器2", "room": "台所", "generation_rate": s_heat["台所"]["機器2"], **ratio["機器"]},
        {"key": "浴室_照明", "room": "浴室", "generation_rate": s_heat["浴室"]["照明"], **ratio["白熱灯"]},
        {"key": "1階トイレ_照明", "room": "1階トイレ", "generation_rate": s_heat["1階トイレ"]["照明"], **ratio["白熱灯"]},
        {"key": "1階トイレ_機器", "room": "1階トイレ", "generation_rate": s_heat["1階トイレ"]["機器"], **ratio["機器"]},
        {"key": "洗面所_照明", "room": "洗面所", "generation_rate": s_heat["洗面所"]["照明"], **ratio["白熱灯"]},
        {"key": "洗面所_機器", "room": "洗面所", "generation_rate": s_heat["洗面所"]["機器"], **ratio["機器"]},
        {"key": "ホール_照明1", "room": "ホール", "generation_rate": s_heat["ホール"]["照明1"], **ratio["白熱灯"]},
        {"key": "ホール_照明2", "room": "ホール", "generation_rate": s_heat["ホール"]["照明2"], **ratio["白熱灯"]},
        {"key": "寝室_人体", "room": "寝室", "generation_rate": s_heat["寝室"]["人体"], **ratio["人体"]},
        {"key": "寝室_照明", "room": "寝室", "generation_rate": s_heat["寝室"]["照明"], **ratio["蛍光灯"]},
        {"key": "寝室_機器", "room": "寝室", "generation_rate": s_heat["寝室"]["機器"], **ratio["機器"]},
        {"key": "子供室1_人体", "room": "子供室1", "generation_rate": s_heat["子供室1"]["人体"], **ratio["人体"]},
        {"key": "子供室1_照明", "room": "子供室1", "generation_rate": s_heat["子供室1"]["照明"], **ratio["蛍光灯"]},
        {"key": "子供室1_機器", "room": "子供室1", "generation_rate": s_heat["子供室1"]["機器"], **ratio["機器"]},
        {"key": "子供室2_人体", "room": "子供室2", "generation_rate": s_heat["子供室2"]["人体"], **ratio["人体"]},
        {"key": "子供室2_照明", "room": "子供室2", "generation_rate": s_heat["子供室2"]["照明"], **ratio["蛍光灯"]},
        {"key": "子供室2_機器", "room": "子供室2", "generation_rate": s_heat["子供室2"]["機器"], **ratio["機器"]},
    ]


def make_humidity_sources() -> list[dict[str, Any]]:
    from vtsimnx.schedule import latent_moisture as h_heat

    return [
        {"key": "LD_人体", "room": "LD", "generation_rate": h_heat["LD"]["人体"]},
        {"key": "台所_機器", "room": "台所", "generation_rate": h_heat["台所"]["機器"]},
        {"key": "寝室_人体", "room": "寝室", "generation_rate": h_heat["寝室"]["人体"]},
        {"key": "子供室1_人体", "room": "子供室1", "generation_rate": h_heat["子供室1"]["人体"]},
        {"key": "子供室2_人体", "room": "子供室2", "generation_rate": h_heat["子供室2"]["人体"]},
    ]

