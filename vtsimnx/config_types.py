from typing import TypedDict, List, Dict, Union, Optional
from enum import Enum
import numpy as np

class NodeTypeEnum(str, Enum):
    AIRCON = "aircon"
    CAPACITY = "capacity"
    NORMAL = "normal"


class NodeSubtypeEnum(str, Enum):
    SURFACE = "surface"
    INTERNAL = "internal"


class VentilationBranchTypeEnum(str, Enum):
    SIMPLE_OPENING = "simple_opening"
    GAP = "gap"
    FAN = "fan"
    FIXED_FLOW = "fixed_flow"


class ThermalBranchTypeEnum(str, Enum):
    CONDUCTANCE = "conductance"
    HEAT_GENERATION = "heat_generation"


class IndexType(TypedDict):
    start: str
    end: str
    timestep: int
    length: int


class ToleranceType(TypedDict):
    ventilation: float
    thermal: float
    convergence: float


class CalcFlagType(TypedDict):
    p: bool
    t: bool
    x: bool
    c: bool


class SimConfigType(TypedDict):
    index: IndexType
    tolerance: ToleranceType
    calc_flag: CalcFlagType


class NodeType(TypedDict):
    key: str
    type: NodeTypeEnum
    subtype: Optional[NodeSubtypeEnum]
    ref_node: Optional[str]
    v: Optional[float]
    beta: Optional[float]
    calc_p: Optional[bool]
    calc_t: Optional[bool]
    calc_x: Optional[bool]
    calc_c: Optional[bool]
    p: Optional[np.ndarray]
    t: Optional[np.ndarray]
    x: Optional[np.ndarray]
    c: Optional[np.ndarray]
    pre_temp: Optional[float]


class VentilationBranchType(TypedDict):
    key: str
    type: VentilationBranchTypeEnum
    source: str
    target: str
    enable: List[bool]
    h_from: float
    h_to: float
    eta: Optional[float]
    alpha: Optional[float]
    area: Optional[float]
    a: Optional[float]
    n: Optional[float]
    p_max: Optional[float]
    q_max: Optional[float]
    p1: Optional[float]
    q1: Optional[float]
    vol: Optional[float]
    humidity_generation: Optional[np.ndarray]
    dust_generation: Optional[np.ndarray]


class ThermalBranchType(TypedDict):
    key: str
    type: ThermalBranchTypeEnum
    source: str
    target: str
    enable: List[bool]
    conductance: Optional[float]
    u_value: Optional[float]
    area: Optional[float]
    heat_generation: Optional[np.ndarray]
