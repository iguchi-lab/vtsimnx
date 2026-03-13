import json
import math
from pathlib import Path

import vtsimnx.archenv.archenv as ae


def _constants():
    root = Path(__file__).resolve().parents[2]
    p = root / "spec" / "archenv_constants.json"
    with p.open(encoding="utf-8") as f:
        return json.load(f)["constants"]


def test_archenv_python_constants_match_registry():
    c = _constants()
    assert ae.P_ATM == c["STANDARD_ATMOSPHERIC_PRESSURE"]["python_value"]
    assert ae.Air_Cp == c["SPECIFIC_HEAT_AIR"]["python_value"]
    assert ae.Vap_Cp == c["SPECIFIC_HEAT_WATER_VAPOR"]["python_value"]
    assert ae.Vap_L == c["LATENT_HEAT_VAPORIZATION"]["python_value"]
    assert ae.Sigma == c["SIGMA"]["python_value"]
    assert ae.Solar_I == c["SOLAR_CONSTANT"]["python_value"]


def test_archenv_python_cpp_crosscheck_with_tolerance():
    c = _constants()
    assert math.isclose(
        ae.Air_Cp * 1000.0,
        c["SPECIFIC_HEAT_AIR"]["cpp_value"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        ae.Vap_Cp * 1000.0,
        c["SPECIFIC_HEAT_WATER_VAPOR"]["cpp_value"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        ae.Vap_L * 1000.0,
        c["LATENT_HEAT_VAPORIZATION"]["cpp_value"],
        rel_tol=0.0,
        abs_tol=1e-12,
    )
