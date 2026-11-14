"""
builder 入力（raw_config）向けの段階的 Pydantic スキーマ。

OpenAPI に主要構造を出しつつ、未知キーは extra="allow" + 明示モードで制御する。
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.units import (
    AREA_M2,
    CONCENTRATION,
    CONDUCTANCE_W_K,
    HEAT_RATE_W,
    HUMIDITY_RATIO,
    LENGTH_M,
    MOISTURE_GEN_KG_S,
    PRESSURE_PA,
    SOLAR_IRRADIANCE_W_M2,
    TEMPERATURE_C,
    THERMAL_MASS_J_K,
    TIME_S,
    U_VALUE_W_M2K,
    VOLUME_FLOW_M3_S,
    VOLUME_M3,
    field_extra,
)


UnknownKeysMode = Literal["strip", "error"]

# JSON 経由の時系列（スカラー or 配列）
ScalarOrSeries = Union[float, int, bool, List[float], List[int], List[bool], str, List[str]]


class _StrictExtraBase(BaseModel):
    """既知フィールドを OpenAPI に出し、未知キーは model_extra に保持する。"""

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class SimulationIndex(_StrictExtraBase):
    start: str = Field(description="シミュレーション開始時刻（ISO8601 等）")
    end: str = Field(description="シミュレーション終了時刻")
    timestep: Union[int, float] = Field(
        description="時間刻み",
        json_schema_extra=field_extra(TIME_S),
    )
    length: int = Field(description="ステップ数")


class SimulationTolerance(_StrictExtraBase):
    ventilation: Optional[float] = Field(
        default=None,
        description="圧力/換気残差許容 [m³/s]",
        json_schema_extra=field_extra(VOLUME_FLOW_M3_S),
    )
    thermal: Optional[float] = Field(
        default=None,
        description="互換代表値（未指定の用途別キーへコピー）。従来は熱/空調で共用",
    )
    convergence: Optional[float] = Field(default=None, description="連成収束許容")
    aircon_temperature: Optional[float] = Field(
        default=None,
        description="空調ON/OFF温度差許容 [K]",
    )
    thermal_balance: Optional[float] = Field(
        default=None,
        description="熱収支RMSE合否許容 [W]",
        json_schema_extra=field_extra(HEAT_RATE_W),
    )
    thermal_linear_residual: Optional[float] = Field(
        default=None,
        description="DirectT 線形残差の相対許容 max|Ax-b|/max(1,|b|)",
    )
    coupling_pressure: Optional[float] = Field(
        default=None,
        description="連成(圧力)許容 [Pa]",
        json_schema_extra=field_extra(PRESSURE_PA),
    )
    coupling_temperature: Optional[float] = Field(
        default=None,
        description="連成(温度)許容 [K]",
    )
    coupling_humidity: Optional[float] = Field(
        default=None,
        description="連成(湿気)許容",
    )


class SimulationCalcFlag(_StrictExtraBase):
    p: Optional[bool] = None
    t: Optional[bool] = None
    x: Optional[bool] = None
    c: Optional[bool] = None


class SimulationLog(_StrictExtraBase):
    verbosity: Optional[int] = Field(default=None, strict=True, ge=0, le=3)


class SimulationIteration(_StrictExtraBase):
    max_inner: Optional[int] = Field(default=None, strict=True, gt=0)
    max_coupling: Optional[int] = Field(default=None, strict=True, gt=0)
    max_aircon_control: Optional[int] = Field(default=None, strict=True, gt=0)


class SimulationCoupling(_StrictExtraBase):
    moisture_enabled: Optional[bool] = None
    humidity_relaxation: Optional[float] = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    temperature_relaxation: Optional[float] = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    latent_relaxation: Optional[float] = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    latent_coupling_mode: Optional[Union[Literal[0, 1, 2], Literal[
        "disabled", "Disabled", "DISABLED", "from_humidity_change", "FromHumidityChange",
        "feedback_to_thermal", "FeedbackToThermal", "from_phase_change", "FromPhaseChange",
    ]]] = None
    latent_absolute_tolerance_w: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    latent_relative_tolerance: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)
    moist_enthalpy_enabled: Optional[bool] = None
    humidity_solver_max_iter: Optional[int] = Field(default=None, strict=True, gt=0)
    humidity_solver_tolerance: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)


class SimulationSection(_StrictExtraBase):
    index: SimulationIndex
    tolerance: Optional[SimulationTolerance] = None
    calc_flag: Optional[SimulationCalcFlag] = None
    log: Optional[SimulationLog] = None
    iteration: Optional[SimulationIteration] = None
    coupling: Optional[SimulationCoupling] = None
    max_inner_iteration: Optional[int] = Field(default=None, strict=True, gt=0)
    max_coupling_iteration: Optional[int] = Field(default=None, strict=True, gt=0)
    max_aircon_control_iteration: Optional[int] = Field(default=None, strict=True, gt=0)


class NodeModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    t: Optional[ScalarOrSeries] = Field(
        default=None, description="温度", json_schema_extra=field_extra(TEMPERATURE_C)
    )
    p: Optional[ScalarOrSeries] = Field(
        default=None, description="圧力", json_schema_extra=field_extra(PRESSURE_PA)
    )
    x: Optional[ScalarOrSeries] = Field(
        default=None,
        description="絶対湿度",
        json_schema_extra=field_extra(HUMIDITY_RATIO),
    )
    c: Optional[ScalarOrSeries] = Field(
        default=None, description="濃度", json_schema_extra=field_extra(CONCENTRATION)
    )
    calc_t: Optional[bool] = None
    calc_p: Optional[bool] = None
    calc_x: Optional[bool] = None
    calc_c: Optional[bool] = None
    v: Optional[float] = Field(
        default=None, description="気積", json_schema_extra=field_extra(VOLUME_M3)
    )
    beta: Optional[ScalarOrSeries] = Field(
        default=None, description="沈着係数", json_schema_extra=field_extra("1/s")
    )
    thermal_mass: Optional[float] = Field(
        default=None,
        description="熱容量",
        json_schema_extra=field_extra(THERMAL_MASS_J_K),
    )
    moisture_capacity: Optional[float] = None
    moisture_capacity_unit: Optional[str] = None
    w: Optional[ScalarOrSeries] = None
    ref_node: Optional[str] = None
    in_node: Optional[str] = None
    set_node: Optional[str] = None
    outside_node: Optional[str] = None
    pre_temp: Optional[ScalarOrSeries] = Field(
        default=None,
        description="エアコン設定温度",
        json_schema_extra=field_extra(TEMPERATURE_C),
    )
    control_mode: Literal["temperature_priority", "humidity_priority", "temperature_humidity"] = "temperature_priority"
    max_overcooling: float = Field(default=2.0, ge=0.0, le=10.0, description="湿度優先時の許容過冷却幅 [K]")
    min_coil_temperature: float = Field(default=5.0, ge=0.0, le=20.0, description="理想除湿コイルの最低温度 [℃]")
    reheat_cop: float = Field(default=1.0, gt=0.0, description="温湿度満足時の再熱COP（既定は電気ヒーター）")
    pre_rh: Optional[ScalarOrSeries] = Field(
        default=None,
        description="室RHの評価・制御目標（%）。温度優先では未達評価のみ、湿度優先・温湿度満足では冷房時必須",
        json_schema_extra=field_extra("%"),
    )
    model: Optional[str] = None
    mode: Optional[ScalarOrSeries] = None
    ac_spec: Optional["AcSpecModel"] = None
    vol_zero: Optional[float] = Field(
        default=None,
        description="熱処理負荷が0のときの循環風量（DUCT_CENTRAL）。未指定は0",
        json_schema_extra=field_extra(VOLUME_FLOW_M3_S),
    )


class VentilationBranchModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    enable: Optional[Union[bool, List[bool]]] = None
    h_from: Optional[float] = Field(
        default=None, description="出発高さ", json_schema_extra=field_extra(LENGTH_M)
    )
    h_to: Optional[float] = Field(
        default=None, description="到達高さ", json_schema_extra=field_extra(LENGTH_M)
    )
    eta: Optional[float] = None
    alpha: Optional[float] = Field(
        default=None, description="有効開口率", json_schema_extra=field_extra("-")
    )
    area: Optional[float] = Field(
        default=None, description="開口面積", json_schema_extra=field_extra(AREA_M2)
    )
    a: Optional[float] = None
    n: Optional[float] = None
    p_max: Optional[float] = Field(
        default=None, description="最大静圧", json_schema_extra=field_extra(PRESSURE_PA)
    )
    q_max: Optional[float] = None
    p1: Optional[float] = None
    q1: Optional[float] = None
    vol: Optional[ScalarOrSeries] = Field(
        default=None,
        description="固定風量（builder/solver 基本単位は m3/s）",
        json_schema_extra=field_extra(VOLUME_FLOW_M3_S),
    )
    k_total: Optional[float] = None
    friction_factor: Optional[float] = None
    lambda_: Optional[float] = Field(default=None, alias="lambda")
    length: Optional[float] = Field(
        default=None, json_schema_extra=field_extra(LENGTH_M)
    )
    diameter: Optional[float] = Field(
        default=None, json_schema_extra=field_extra(LENGTH_M)
    )
    zeta_total: Optional[float] = None
    humidity_generation: Optional[ScalarOrSeries] = Field(
        default=None,
        description="発湿",
        json_schema_extra=field_extra(MOISTURE_GEN_KG_S),
    )
    dust_generation: Optional[ScalarOrSeries] = None


class ThermalBranchModel(_StrictExtraBase):
    key: str
    type: Optional[str] = None
    subtype: Optional[str] = None
    comment: Optional[str] = None
    source: Optional[str] = None
    target: Optional[str] = None
    enable: Optional[Union[bool, List[bool]]] = None
    conductance: Optional[float] = Field(
        default=None,
        description="熱コンダクタンス",
        json_schema_extra=field_extra(CONDUCTANCE_W_K),
    )
    u_value: Optional[float] = Field(
        default=None,
        description="熱貫流率",
        json_schema_extra=field_extra(U_VALUE_W_M2K),
    )
    area: Optional[float] = Field(
        default=None, description="面積", json_schema_extra=field_extra(AREA_M2)
    )
    heat_generation: Optional[ScalarOrSeries] = Field(
        default=None,
        description="発熱",
        json_schema_extra=field_extra(HEAT_RATE_W),
    )
    resp_a_src: Optional[List[float]] = None
    resp_b_src: Optional[List[float]] = None
    resp_c_src: Optional[List[float]] = None
    resp_a_tgt: Optional[List[float]] = None
    resp_b_tgt: Optional[List[float]] = None
    resp_c_tgt: Optional[List[float]] = None
    moisture_conductance: Optional[float] = None
    moisture_transfer_type: Optional[Literal[
        "phase_change",
        "vapor_diffusion",
        "liquid_transport",
        "sorption",
    ]] = Field(
        default=None,
        description=(
            "湿気移動種別。phase_change のみ潜熱診断対象。"
            "liquid_transport / sorption は将来用分類（現状は kΔx のみ）。"
        ),
    )


class SurfaceLayerModel(_StrictExtraBase):
    lambda_: Optional[float] = Field(
        default=None,
        alias="lambda",
        description="熱伝導率",
        json_schema_extra=field_extra("W/(m·K)"),
    )
    t: Optional[float] = Field(
        default=None, description="厚さ", json_schema_extra=field_extra(LENGTH_M)
    )
    v_capa: Optional[float] = Field(
        default=None,
        description="容積比熱",
        json_schema_extra=field_extra("J/(m3·K)"),
    )
    air_layer: Optional[bool] = None
    ventilated_air_layer: Optional[bool] = None
    thermal_resistance: Optional[float] = Field(
        default=None,
        description="熱抵抗",
        json_schema_extra=field_extra("m2·K/W"),
    )
    r_value: Optional[float] = None
    r: Optional[float] = None
    air_v_capa: Optional[float] = None
    alpha_c1: Optional[float] = None
    alpha_c2: Optional[float] = None
    alpha_r: Optional[float] = None


class SurfaceModel(_StrictExtraBase):
    key: str
    part: Optional[str] = None
    area: Optional[float] = Field(
        default=None, description="面積", json_schema_extra=field_extra(AREA_M2)
    )
    layers: Optional[List[SurfaceLayerModel]] = None
    u_value: Optional[float] = Field(
        default=None,
        description="熱貫流率",
        json_schema_extra=field_extra(U_VALUE_W_M2K),
    )
    alpha_i: Optional[float] = None
    alpha_o: Optional[float] = None
    layer_method: Optional[str] = None
    solar: Optional[ScalarOrSeries] = Field(
        default=None,
        description="日射取得",
        json_schema_extra=field_extra(SOLAR_IRRADIANCE_W_M2),
    )
    nocturnal: Optional[ScalarOrSeries] = Field(
        default=None,
        description="夜間放射（推奨キー）",
        json_schema_extra=field_extra(SOLAR_IRRADIANCE_W_M2),
    )
    night_radiation: Optional[ScalarOrSeries] = Field(
        default=None,
        description="夜間放射（互換キー）",
        json_schema_extra=field_extra(SOLAR_IRRADIANCE_W_M2),
    )
    eta: Optional[float] = None
    epsilon: Optional[float] = None
    comment: Optional[str] = None
    a_capacity: Optional[float] = None
    response_method: Optional[str] = None
    response_terms: Optional[int] = None
    SCC: Optional[float] = None
    SCR: Optional[float] = None
    scc: Optional[float] = None
    scr: Optional[float] = None
    response: Optional[Dict[str, Any]] = None


POWER_KW = "kW"


class CapacityRatingModel(_StrictExtraBase):
    """min/mid/rtd/max/dsgn。Q/P では kW、V_inner/V_outer では m³/s。"""

    min: Optional[float] = None
    mid: Optional[float] = None
    rtd: Optional[float] = None
    max: Optional[float] = None
    dsgn: Optional[float] = Field(default=None, description="設計値（V_inner.dsgn は m³/s）")


class ModeCapacityModel(_StrictExtraBase):
    cooling: Optional[CapacityRatingModel] = None
    heating: Optional[CapacityRatingModel] = None


class MinProcessModel(_StrictExtraBase):
    heating: Optional[float] = Field(
        default=None,
        description="制御用最低処理能力",
        json_schema_extra=field_extra(POWER_KW),
    )
    cooling: Optional[float] = Field(
        default=None,
        description="制御用最低処理能力",
        json_schema_extra=field_extra(POWER_KW),
    )


class AcSpecModel(_StrictExtraBase):
    """
    空調機器仕様。カタログ値（Q/P/V_*）と制御用 min_process を分離する。
    モデル種別ごとの必須キーは OpenAPI description と aircon_spec_reference.md を参照。
    """

    Q: Optional[ModeCapacityModel] = Field(default=None, description="能力 [kW]")
    P: Optional[ModeCapacityModel] = Field(default=None, description="消費電力 [kW]")
    P_fan: Optional[ModeCapacityModel] = Field(default=None, description="ファン消費電力 [kW]")
    V_inner: Optional[ModeCapacityModel] = Field(default=None, description="室内機風量 [m³/s]")
    V_outer: Optional[ModeCapacityModel] = Field(default=None, description="室外機風量 [m³/s]")
    min_process: Optional[MinProcessModel] = Field(
        default=None,
        description="制御用最低処理能力 [kW]。カタログ Q.min とは独立",
    )
    latent_method: Optional[str] = Field(
        default=None,
        description="潜熱処理方式: rh95 / bf / coil_aoaf / none",
    )
    bf: Optional[float] = None
    BF: Optional[float] = None
    bypass_factor: Optional[float] = None
    Af: Optional[float] = Field(default=None, json_schema_extra=field_extra(AREA_M2))
    coil_face_area: Optional[float] = Field(default=None, json_schema_extra=field_extra(AREA_M2))
    Ao: Optional[float] = Field(default=None, json_schema_extra=field_extra(AREA_M2))
    coil_surface_area: Optional[float] = Field(default=None, json_schema_extra=field_extra(AREA_M2))
    dualcompressor: Optional[bool] = None
    max_heat_capacity: Optional[float] = Field(
        default=None,
        description="最大処理熱量（互換）",
        json_schema_extra=field_extra(HEAT_RATE_W),
    )


class HeatSourceModel(_StrictExtraBase):
    key: Optional[str] = Field(default=None, description="ログ用識別名")
    room: str = Field(description="対象室ノード（nodes[].key）")
    generation_rate: ScalarOrSeries = Field(
        description="発熱量（顕熱）",
        json_schema_extra=field_extra(HEAT_RATE_W),
    )
    convection: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="対流成分比。未指定かつ radiation 未指定なら 1",
    )
    radiation: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="放射成分比。未指定かつ convection 未指定なら 0",
    )

    @model_validator(mode="after")
    def _fractions(self) -> "HeatSourceModel":
        if self.convection is not None and self.radiation is not None:
            if (self.convection + self.radiation) > 1.0 + 1e-9:
                raise ValueError("heat_source.convection + radiation は 1 以下である必要があります")
        return self


class HumiditySourceModel(_StrictExtraBase):
    key: Optional[str] = Field(default=None, description="ログ用識別名")
    room: Optional[str] = Field(default=None, description="対象室ノード（推奨キー）")
    target_node: Optional[str] = Field(default=None, description="対象室（room の別名）")
    generation_rate: Optional[ScalarOrSeries] = Field(
        default=None,
        description="発湿量",
        json_schema_extra=field_extra(MOISTURE_GEN_KG_S),
    )
    set: Optional[str] = Field(default=None, description="対象室（旧形式）")
    mx: Optional[ScalarOrSeries] = Field(
        default=None,
        description="発湿量（旧形式）",
        json_schema_extra=field_extra(MOISTURE_GEN_KG_S),
    )

    @model_validator(mode="after")
    def _require_target_and_rate(self) -> "HumiditySourceModel":
        room = self.room or self.target_node or self.set
        generation_rate = self.generation_rate if self.generation_rate is not None else self.mx
        if generation_rate is not None and room:
            if room != self.room or generation_rate != self.generation_rate:
                return self.model_copy(update={"room": room, "generation_rate": generation_rate})
            return self
        raise ValueError(
            "humidity_source には room（または target_node）と generation_rate、"
            "あるいは旧形式の set と mx が必要です"
        )


class AirconModel(_StrictExtraBase):
    key: str
    set: Optional[str] = None
    outside: Optional[str] = None
    pre_temp: Optional[ScalarOrSeries] = Field(
        default=None,
        description="設定温度",
        json_schema_extra=field_extra(TEMPERATURE_C),
    )
    control_mode: Literal["temperature_priority", "humidity_priority", "temperature_humidity"] = "temperature_priority"
    max_overcooling: float = Field(default=2.0, ge=0.0, le=10.0, description="湿度優先時の許容過冷却幅 [K]")
    min_coil_temperature: float = Field(default=5.0, ge=0.0, le=20.0, description="理想除湿コイルの最低温度 [℃]")
    reheat_cop: float = Field(default=1.0, gt=0.0, description="温湿度満足時の再熱COP（既定は電気ヒーター）")
    pre_rh: Optional[ScalarOrSeries] = Field(
        default=None,
        description="室RHの評価・制御目標（%）。温度優先では未達評価のみ、湿度優先・温湿度満足では冷房時必須",
        json_schema_extra=field_extra("%"),
    )
    mode: Optional[ScalarOrSeries] = None
    model: Optional[str] = Field(
        default=None,
        description="CRIEPI / RAC / DUCT_CENTRAL / LATENT_EVALUATE。未指定時 RAC",
    )
    ac_spec: Optional[AcSpecModel] = None
    vol: Optional[ScalarOrSeries] = Field(
        default=None,
        description="送風量",
        json_schema_extra=field_extra(VOLUME_FLOW_M3_S),
    )
    vol_zero: Optional[float] = Field(
        default=None,
        description="熱処理負荷が0のときの循環風量（DUCT_CENTRAL）。未指定は0",
        json_schema_extra=field_extra(VOLUME_FLOW_M3_S),
    )
    p_max: Optional[float] = Field(
        default=None,
        description="空調ファン定格の最大静圧。指定時は固定風量ではなく PQ",
        json_schema_extra=field_extra(PRESSURE_PA),
    )
    p1: Optional[float] = None
    q1: Optional[float] = None
    q_max: Optional[float] = None
    area: Optional[float] = Field(
        default=None,
        description="PQ 時の空調→吹出つなぎの断面積",
        json_schema_extra=field_extra(AREA_M2),
    )
    k_total: Optional[float] = None
    in_: Optional[str] = Field(default=None, alias="in")
    out: Optional[str] = None


class HeatRecoveryFanModel(_StrictExtraBase):
    p_max: Optional[float] = None
    p1: Optional[float] = None
    q1: Optional[float] = None
    q_max: Optional[float] = None


class HeatRecoveryVentModel(_StrictExtraBase):
    key: str
    outdoor: Optional[str] = None
    room: Optional[str] = None
    set: Optional[str] = None
    oa: Optional[str] = None
    sa: Optional[str] = None
    ra: Optional[str] = None
    ea: Optional[str] = None
    in_: Optional[str] = Field(default=None, alias="in")
    out: Optional[str] = None
    return_: Optional[str] = Field(default=None, alias="return")
    exhaust_out: Optional[str] = None
    recovery: Optional[str] = None
    model: Optional[str] = None
    eta_t: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    eta_x: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    vol: Optional[float] = None
    area: Optional[float] = None
    k_total: Optional[float] = None
    supply: Optional[HeatRecoveryFanModel] = None
    exhaust: Optional[HeatRecoveryFanModel] = None


class BuilderOptions(_StrictExtraBase):
    surface_layer_method: Optional[str] = None
    response_method: Optional[str] = None
    response_terms: Optional[int] = None
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_heat_recovery_vent: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None


class RawSimConfig(_StrictExtraBase):
    """API / builder が受け取る raw_config。"""

    simulation: SimulationSection
    nodes: List[NodeModel]
    ventilation_branches: List[VentilationBranchModel] = Field(default_factory=list)
    thermal_branches: List[ThermalBranchModel] = Field(default_factory=list)
    builder: Optional[BuilderOptions] = None
    surfaces: Optional[List[SurfaceModel]] = None
    aircon: Optional[List[AirconModel]] = None
    heat_recovery_vent: Optional[List[HeatRecoveryVentModel]] = None
    heat_source: Optional[List[HeatSourceModel]] = None
    humidity_source: Optional[List[HumiditySourceModel]] = None
    # 互換トップレベル builder オプション
    surface_layer_method: Optional[str] = None
    response_method: Optional[str] = None
    response_terms: Optional[int] = None
    add_surface: Optional[bool] = None
    add_aircon: Optional[bool] = None
    add_heat_recovery_vent: Optional[bool] = None
    add_capacity: Optional[bool] = None
    add_moisture_capacity: Optional[bool] = None
    add_surface_solar: Optional[bool] = None
    add_surface_nocturnal: Optional[bool] = None
    add_surface_radiation: Optional[bool] = None
    add_surface_radiation_exclude_glass: Optional[bool] = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_source_aliases(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if "heat_source" not in data and "heat_sources" in data:
            data["heat_source"] = data.pop("heat_sources")
        else:
            data.pop("heat_sources", None)
        if "humidity_source" not in data and "humidity_sources" in data:
            data["humidity_source"] = data.pop("humidity_sources")
        else:
            data.pop("humidity_sources", None)

        hs = data.get("humidity_source")
        if isinstance(hs, dict):
            converted: List[Dict[str, Any]] = []
            for key, item in hs.items():
                if not isinstance(item, dict):
                    converted.append({"key": str(key)})
                    continue
                row = {"key": str(key), **item}
                if row.get("room") is None and row.get("set"):
                    row["room"] = row["set"]
                if row.get("generation_rate") is None and "mx" in row:
                    row["generation_rate"] = row["mx"]
                converted.append(row)
            data["humidity_source"] = converted
        return data


class UnknownFieldError(ValueError):
    """未知キーを error モードで拒否したとき（HTTP 422 向け）。"""

    def __init__(self, message: str, *, details: Optional[List[Dict[str, Any]]] = None) -> None:
        super().__init__(message)
        self.details = details or []


def format_field_path(parts: Sequence[Any]) -> str:
    """['heat_source', 0, 'foo'] を heat_source[0].foo にする。"""
    out = ""
    for part in parts:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            token = str(part)
            out = f"{out}.{token}" if out else token
    return out or "config"


def collect_unknown_fields(model: BaseModel, *, path: tuple[Any, ...] = ()) -> List[Dict[str, Any]]:
    """Pydantic model_extra を再帰走査して未知キー一覧を返す。"""
    found: List[Dict[str, Any]] = []
    extras = getattr(model, "__pydantic_extra__", None) or {}
    for key in extras:
        found.append({"path": list(path), "field": str(key)})

    for name in type(model).model_fields:
        value = getattr(model, name, None)
        child = path + (name,)
        if isinstance(value, BaseModel):
            found.extend(collect_unknown_fields(value, path=child))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    found.extend(collect_unknown_fields(item, path=child + (i,)))
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
                "loc": ["config", *u["path"], u["field"]],
                "msg": (
                    f"{format_field_path(u['path'])} に未定義のフィールド "
                    f"'{u['field']}' があります"
                ),
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
        ctx = format_field_path(u["path"])
        msg = f"{ctx} に未定義のフィールド '{u['field']}' が指定されました。無視しました。"
        warnings.append(msg)
        warning_details.append(
            {
                "code": "unknown_field_stripped",
                "message": msg,
                "context": ctx,
                "field": u["field"],
                "path": list(u["path"]) + [u["field"]],
                "severity": "warning",
            }
        )

    data = dump_config_without_extras(model)
    assert isinstance(data, dict)
    _normalize_aliases(data)
    return data, warnings, warning_details


__all__ = [
    "AcSpecModel",
    "AirconModel",
    "BuilderOptions",
    "HeatSourceModel",
    "HumiditySourceModel",
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
    "format_field_path",
    "prepare_raw_config",
]
