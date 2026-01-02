import vtsimnx as vt


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


