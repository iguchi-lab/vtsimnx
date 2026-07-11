"""
builder 入力（raw_config）向けの段階的 Pydantic スキーマ。

OpenAPI に主要構造を出しつつ、未知キーは extra="allow" + 明示モードで制御する。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


UnknownKeysMode = Literal["strip", "error"]

# JSON 経由の時系列（スカラー or 配列）
ScalarOrSeries = Union[float, int, bool, List[float], List[int], List[bool], str, List[str]]


class _StrictExtraBase(BaseModel):
    """既知フィールドを OpenAPI に出し、未知キーは model_extra に保持する。"""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class SimulationIndex(_StrictExtraBase):
    start: str
    end: str
    timestep: Union[int, float]
    length: int


class SimulationTolerance(_StrictExtraBase):
    ventilation: Optional[float] = None
    thermal: Optional[float] = None
    convergence: Optional[float] = None


class SimulationCalcFlag(_StrictExtraBase):
    p: Optional[bool] = None
    t: Optional[bool] = None
    x: Optional[bool] = None
    c: Optional[bool] = None


class SimulationLog(_StrictExtraBase):
    verbosity: Optional[int] = None


class SimulationSection(_StrictExtraBase):
    index: SimulationIndex
    tolerance: Optional[SimulationTolerance] = None
    calc_flag: Optional[SimulationCalcFlag] = None
    log: Optional[SimulationLog] = None


class NodeModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    t: Optional[ScalarOrSeries] = None
    p: Optional[ScalarOrSeries] = None
    x: Optional[ScalarOrSeries] = None
    c: Optional[ScalarOrSeries] = None
    calc_t: Optional[bool] = None
    calc_p: Optional[bool] = None
    calc_x: Optional[bool] = None
    calc_c: Optional[bool] = None
    v: Optional[float] = None
    beta: Optional[ScalarOrSeries] = None
    thermal_mass: Optional[float] = None
    moisture_capacity: Optional[float] = None
    moisture_capacity_unit: Optional[str] = None
    w: Optional[ScalarOrSeries] = None
    ref_node: Optional[str] = None
    in_node: Optional[str] = None
    set_node: Optional[str] = None
    outside_node: Optional[str] = None
    pre_temp: Optional[ScalarOrSeries] = None
    model: Optional[str] = None
    mode: Optional[ScalarOrSeries] = None
    ac_spec: Optional[Dict[str, Any]] = None


class VentilationBranchModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    enable: Optional[Union[bool, List[bool]]] = None
    h_from: Optional[float] = None
    h_to: Optional[float] = None
    eta: Optional[ScalarOrSeries] = None
    alpha: Optional[float] = None
    area: Optional[float] = None
    a: Optional[float] = None
    n: Optional[float] = None
    p_max: Optional[float] = None
    q_max: Optional[float] = None
    p1: Optional[float] = None
    q1: Optional[float] = None
    vol: Optional[ScalarOrSeries] = None
    k_total: Optional[float] = None
    friction_factor: Optional[float] = None
    lambda_: Optional[float] = Field(default=None, alias="lambda")
    length: Optional[float] = None
    diameter: Optional[float] = None
    zeta_total: Optional[float] = None
    humidity_generation: Optional[ScalarOrSeries] = None
    dust_generation: Optional[ScalarOrSeries] = None


class ThermalBranchModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    enable: Optional[Union[bool, List[bool]]] = None
    conductance: Optional[float] = None
    u_value: Optional[float] = None
    area: Optional[float] = None
    heat_generation: Optional[ScalarOrSeries] = None
    resp_a_src: Optional[List[float]] = None
    resp_b_src: Optional[List[float]] = None
    resp_c_src: Optional[List[float]] = None
    resp_a_tgt: Optional[List[float]] = None
    resp_b_tgt: Optional[List[float]] = None
    resp_c_tgt: Optional[List[float]] = None
    moisture_conductance: Optional[float] = None


class SurfaceLayerModel(_StrictExtraBase):
    lambda_: Optional[float] = Field(default=None, alias="lambda")
    t: Optional[float] = None
    v_capa: Optional[float] = None
    air_layer: Optional[bool] = None
    ventilated_air_layer: Optional[bool] = None
    thermal_resistance: Optional[float] = None
    r_value: Optional[float] = None
    r: Optional[float] = None
    air_v_capa: Optional[float] = None
    alpha_c1: Optional[float] = None
    alpha_c2: Optional[float] = None
    alpha_r: Optional[float] = None


class SurfaceModel(_StrictExtraBase):
    key: str
    part: Optional[str] = None
    area: Optional[float] = None
    layers: Optional[List[SurfaceLayerModel]] = None
    u_value: Optional[float] = None
    alpha_i: Optional[float] = None
    alpha_o: Optional[float] = None
    layer_method: Optional[str] = None
    solar: Optional[ScalarOrSeries] = None
    eta: Optional[float] = None
    epsilon: Optional[float] = None
    response: Optional[Dict[str, Any]] = None


class AirconModel(_StrictExtraBase):
    key: str
    set: Optional[str] = None
    outside: Optional[str] = None
    pre_temp: Optional[ScalarOrSeries] = None
    mode: Optional[ScalarOrSeries] = None
    model: Optional[str] = None
    ac_spec: Optional[Dict[str, Any]] = None
    vol: Optional[ScalarOrSeries] = None
    in_: Optional[str] = Field(default=None, alias="in")
    out: Optional[str] = None


class BuilderOptions(_StrictExtraBase):
    surface_layer_method: Optional[str] = None
    response_method: Optional[str] = None
    response_terms: Optional[int] = None
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None


class RawSimConfig(_StrictExtraBase):
    """
    API / builder が受け取る raw_config。
    heat_source / humidity_source は当面ゆるい dict リスト。
    """

    simulation: SimulationSection
    nodes: List[NodeModel]
    ventilation_branches: List[VentilationBranchModel] = Field(default_factory=list)
    thermal_branches: List[ThermalBranchModel] = Field(default_factory=list)
    builder: Optional[BuilderOptions] = None
    surfaces: Optional[List[SurfaceModel]] = None
    aircon: Optional[List[AirconModel]] = None
    heat_source: Optional[List[Dict[str, Any]]] = None
    humidity_source: Optional[List[Dict[str, Any]]] = None
    # 互換トップレベル builder オプション
    surface_layer_method: Optional[str] = None
    response_method: Optional[str] = None
    response_terms: Optional[int] = None
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None


class UnknownFieldError(ValueError):
    """未知キーを error モードで拒否したとき（HTTP 422 向け）。"""

    def __init__(self, message: str, *, details: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.details = details or []


def collect_unknown_fields(model: BaseModel, *, path: str = "") -> List[Dict[str, str]]:
    """Pydantic model_extra を再帰走査して未知キー一覧を返す。"""
    found: List[Dict[str, str]] = []
    extras = getattr(model, "__pydantic_extra__", None) or {}
    ctx = path or "config"
    for key in extras:
        found.append({"path": ctx, "field": str(key)})

    for name in type(model).model_fields:
        value = getattr(model, name, None)
        child = f"{path}.{name}" if path else name
        if isinstance(value, BaseModel):
            found.extend(collect_unknown_fields(value, path=child))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    found.extend(collect_unknown_fields(item, path=f"{child}[{i}]"))
    return found


def dump_config_without_extras(model: BaseModel) -> Any:
    """model_extra を含めず builder 向け dict を作る。"""
    if isinstance(model, list):  # pragma: no cover
        return [dump_config_without_extras(x) if isinstance(x, BaseModel) else x for x in model]

    out: Dict[str, Any] = {}
    for name, finfo in type(model).model_fields.items():
        key = finfo.alias or name
        # friction_factor は alias が二重定義されうるので、serialization_alias を優先
        ser_alias = getattr(finfo, "serialization_alias", None)
        if ser_alias:
            key = ser_alias
        val = getattr(model, name)
        if val is None:
            # 未設定の Optional は省略（builder 互換）
            if name not in model.model_fields_set:
                continue
            # 明示的 null も省略
            continue
        if isinstance(val, BaseModel):
            out[key] = dump_config_without_extras(val)
        elif isinstance(val, list):
            out[key] = [
                dump_config_without_extras(item) if isinstance(item, BaseModel) else item for item in val
            ]
        else:
            out[key] = val
    return out


def _normalize_aliases(data: Dict[str, Any]) -> None:
    """dump 後の alias フィールドを builder が期待するキーへ寄せる。"""
    vents = data.get("ventilation_branches")
    if isinstance(vents, list):
        for branch in vents:
            if not isinstance(branch, dict):
                continue
            if "lambda" in branch and "friction_factor" not in branch:
                branch["friction_factor"] = branch.pop("lambda")
            elif "lambda" in branch:
                branch.pop("lambda", None)

    surfaces = data.get("surfaces")
    if isinstance(surfaces, list):
        for surface in surfaces:
            if not isinstance(surface, dict):
                continue
            layers = surface.get("layers")
            if not isinstance(layers, list):
                continue
            for layer in layers:
                if isinstance(layer, dict) and "lambda_" in layer:
                    # should not happen if alias dump works; keep safe
                    layer.setdefault("lambda", layer.pop("lambda_"))


def prepare_raw_config(
    config: Union[RawSimConfig, Dict[str, Any]],
    *,
    unknown_keys: UnknownKeysMode = "strip",
) -> tuple[Dict[str, Any], List[str], List[Dict[str, Any]]]:
    """
    RawSimConfig / dict を builder 向け dict に変換する。
    unknown_keys=strip: extras を落とし warning を返す
    unknown_keys=error: extras があれば UnknownFieldError
    """
    if isinstance(config, dict):
        model = RawSimConfig.model_validate(config)
    else:
        model = config

    unknowns = collect_unknown_fields(model)
    if unknowns and unknown_keys == "error":
        details = [
            {
                "type": "unknown_field",
                "loc": ["config", u["path"], u["field"]],
                "msg": f"{u['path']} に未定義のフィールド '{u['field']}' があります",
                "input": u["field"],
            }
            for u in unknowns
        ]
        raise UnknownFieldError(
            "unknown fields in config",
            details=details,
        )

    warnings: List[str] = []
    warning_details: List[Dict[str, Any]] = []
    for u in unknowns:
        msg = f"{u['path']} に未定義のフィールド '{u['field']}' が指定されました。無視しました。"
        warnings.append(msg)
        warning_details.append(
            {
                "code": "unknown_field_stripped",
                "message": msg,
                "context": u["path"],
                "field": u["field"],
            }
        )

    data = dump_config_without_extras(model)
    assert isinstance(data, dict)
    _normalize_aliases(data)
    return data, warnings, warning_details


__all__ = [
    "AirconModel",
    "BuilderOptions",
    "NodeModel",
    "RawSimConfig",
    "SimulationCalcFlag",
    "SimulationIndex",
    "SimulationLog",
    "SimulationSection",
    "SimulationTolerance",
    "SurfaceModel",
    "ThermalBranchModel",
    "UnknownFieldError",
    "UnknownKeysMode",
    "VentilationBranchModel",
    "collect_unknown_fields",
    "dump_config_without_extras",
    "prepare_raw_config",
]
