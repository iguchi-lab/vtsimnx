import numpy as np

import vtsimnx as vt
from vtsimnx.archenv import Vap_L
from vtsimnx.schedule import aircon as aircon_mod
from vtsimnx.schedule.sensible_heat import (
    OCCUPANT_LATENT_AT_24C_W,
    OCCUPANT_SENSIBLE_AT_24C_W,
    OCCUPANT_TOTAL_W,
    occupant_sensible_latent_w,
)


def test_schedule_vol_heat_match_legacy_make_8760_data():
    # vol
    legacy_vol_ld = vt.schedule.make_8760_data(
        [1] * 365,
        vt.schedule.holiday,
        vt.schedule.vent_profiles["LD"]["平日"],
        vt.schedule.vent_profiles["LD"]["休日"],
        [0] * 24,
        [0] * 24,
        0.0,
    )
    assert vt.schedule.build_vol_schedule()["LD"] == legacy_vol_ld
    assert len(vt.schedule.vol["LD"]) == 8760

    # sensible_heat（平日/休日で別プロファイル）
    built = vt.schedule.build_sensible_heat_schedule()
    assert set(built["LD"].keys()) == set(vt.schedule.sensible_heat_profiles["LD"].keys())

    for use_name, profs in vt.schedule.sensible_heat_profiles["LD"].items():
        legacy = vt.schedule.make_8760_data(
            [1] * 365,
            vt.schedule.holiday,
            profs["平日"],
            profs["休日"],
            [0] * 24,
            [0] * 24,
            0.0,
        )
        assert built["LD"][use_name] == legacy
        assert len(vt.schedule.sensible_heat["LD"][use_name]) == 8760

    # 平日 7時は 1人。作業強度2・24℃の顕熱 62 W。
    assert vt.schedule.sensible_heat_profiles["LD"]["人体"]["平日"][6] == OCCUPANT_SENSIBLE_AT_24C_W
    latent_1 = OCCUPANT_LATENT_AT_24C_W / (Vap_L * 1000.0)
    np.testing.assert_allclose(
        vt.schedule.latent_moisture_profiles["LD"]["人体"]["平日"][6], latent_1
    )


def test_schedule_aircon_match_legacy_make_8760_data():
    region = "region1"
    period = vt.schedule.period_1
    holiday_days = vt.schedule.holiday

    # ac_mode
    for room, prof in aircon_mod.ac_mode_profiles.items():
        legacy = vt.schedule.make_8760_data(
            period,
            holiday_days,
            prof["暖房"]["平日"],
            prof["暖房"]["休日"],
            prof["冷房"]["平日"],
            prof["冷房"]["休日"],
            aircon_mod.AC_MODE_STOP,
        )
        assert vt.schedule.build_ac_mode()[region][room] == legacy
        assert len(vt.schedule.ac_mode[region][room]) == 8760

    # pre_tmp
    for room, prof in aircon_mod.pre_tmp_profiles.items():
        legacy = vt.schedule.make_8760_data(
            period,
            holiday_days,
            prof["暖房"]["平日"],
            prof["暖房"]["休日"],
            prof["冷房"]["平日"],
            prof["冷房"]["休日"],
            20.0,
        )
        assert vt.schedule.build_pre_tmp()[region][room] == legacy
        assert len(vt.schedule.pre_tmp[region][room]) == 8760

    # pre_rh
    for room, prof in aircon_mod.rh_profiles.items():
        legacy = vt.schedule.make_8760_data(
            period,
            holiday_days,
            prof["暖房"]["平日"],
            prof["暖房"]["休日"],
            prof["冷房"]["平日"],
            prof["冷房"]["休日"],
            60.0,
        )
        assert vt.schedule.build_pre_rh()[region][room] == legacy
        assert len(vt.schedule.pre_rh[region][room]) == 8760


def test_whole_house_aircon_is_24h_in_season():
    key = aircon_mod.WHOLE_HOUSE_KEY
    for season, mode in (("暖房", aircon_mod.AC_MODE_HEATING), ("冷房", aircon_mod.AC_MODE_COOLING)):
        for day_type in ("平日", "休日"):
            prof = aircon_mod.ac_mode_profiles[key][season][day_type]
            assert prof == [mode] * 24

    # region6: 暖房日は全日 HEATING、冷房日は全日 COOLING、中間期は STOP
    period = vt.schedule.period_6
    series = vt.schedule.ac_mode["region6"][key]
    assert len(series) == 8760
    for day, season in enumerate(period):
        day_vals = series[day * 24 : (day + 1) * 24]
        if season == 1:
            assert day_vals == [aircon_mod.AC_MODE_HEATING] * 24
        elif season == -1:
            assert day_vals == [aircon_mod.AC_MODE_COOLING] * 24
        else:
            assert day_vals == [aircon_mod.AC_MODE_STOP] * 24

    assert vt.schedule.pre_tmp["region6"][key][0] == 20.0
    # 冷房期の先頭日（period_6 で最初の -1）
    cool_day = period.index(-1)
    assert vt.schedule.pre_tmp["region6"][key][cool_day * 24] == 27.0


def test_occupant_heat_is_activity_level_2():
    assert OCCUPANT_TOTAL_W == 106.0
    assert OCCUPANT_SENSIBLE_AT_24C_W == 62.0
    assert OCCUPANT_LATENT_AT_24C_W == 44.0
    sh, lh = occupant_sensible_latent_w(24.0)
    assert (sh, lh) == (62.0, 44.0)
    sh, lh = occupant_sensible_latent_w(24.0 - 44.0 / 3.6)
    np.testing.assert_allclose(sh, 106.0)
    np.testing.assert_allclose(lh, 0.0, atol=1e-12)
    sh, lh = occupant_sensible_latent_w(0.0)
    assert (sh, lh) == (106.0, 0.0)




def test_bri_heat_and_moisture_use_same_occupancy():
    from vtsimnx.schedule.latent_moisture import LATENT_HEAT_J_PER_KG

    sensible = vt.schedule.build_sensible_heat_schedule(occupant_model="bri")["LD"]
    moisture = vt.schedule.build_latent_moisture_schedule(occupant_model="bri")["LD"]
    people = np.asarray(vt.schedule.sensible_heat["LD"]["人体"]) / 62
    np.testing.assert_allclose(sensible["人体"], people * 63)
    np.testing.assert_allclose(np.asarray(moisture["人体"]) * LATENT_HEAT_J_PER_KG, people * 56)
    assert sensible["照明"] == vt.schedule.sensible_heat["LD"]["照明"]
    assert occupant_sensible_latent_w(25, occupant_model="bri") == (59, 60)
