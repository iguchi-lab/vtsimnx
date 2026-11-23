# -*- coding: utf-8 -*-
!pip install -U git+https://github.com/iguchi-lab/vtsimnx/
import vtsimnx as vt

import os

import pandas as pd

"""#準備

##外部ファイルの読み込み
"""

# 気象データは同梱しません。利用条件を確認したHASPファイルを別途用意してください。
weather_file = os.environ.get("VTSIMNX_WEATHER_FILE")
if not weather_file:
    raise ValueError("VTSIMNX_WEATHER_FILE にHASP気象ファイルのパスを設定してください。")
df_i = vt.read_hasp(weather_file)

"""##日射・夜間放射

建物側の方位・傾斜・庇は `vt.standard_house` の部品。直達・拡散・夜間放射は気象から計算する。
"""

h = vt.standard_house

# 壁・屋根に加え、庇付き窓も載せる。LD南は東西2窓の平均。
solar = h.compute_solar(df_i["直達日射量"], df_i["水平面拡散日射量"])
nocturnal = h.compute_nocturnal(df_i["夜間放射量"])

"""#地盤温度"""

df_i["地下1m温度"] = vt.ground_temperature_by_depth(
    depth_m=1.0,
    t_out=df_i["外気温"],
    solar_horizontal=solar["日射熱取得量（水平上向き）"],
    nocturnal_horizontal=df_i["夜間放射量"],
    deep_layer_depth_m=10.0,
    deep_layer_temp_c=10.0,
    thermal_conductivity_w_mk=1.5,
    volumetric_heat_capacity_j_m3k=2.2e6,
    solar_to_surface_temp_coeff=0.003,
    nocturnal_to_surface_temp_coeff=0.003,
    spinup=True,
)

"""#input_dataの作成

部品は `h.available()` で一覧できる。換気は `h.ventilation["24h"]`、空調は `h.aircon["ac4"]` を選んだあと、リストを編集して差し替える。
"""

room_volume = h.rooms["default"]
layers = h.layers["default"]
surface = h.bind_surface_types(
    h.surface_types["default"], layers, solar=solar, nocturnal=nocturnal
)

input_data = {}

"""##シミュレーションの設定"""

input_data["simulation"] = {
    "index": df_i.index,
    "warmup": {"start": h.WARMUP_START},  # 12/15から助走し、結果は1/1から
    "tolerance": {
        "ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6
    },
}

"""##ノード"""

input_data["nodes"] = [
    {"key": "外部", "t": df_i["外気温"], "x": df_i["外気絶対湿度"] / 1000},
    {"key": "地下1m", "t": df_i["地下1m温度"]},
    *h.room_nodes(room_volume),
]

"""##換気回路網"""

# simheat2の個別空調ケースと同じ局所換気経路を使う。
# 外部→ホール→洗面所→浴室→外部、外部→ホール→1階トイレ→外部。
# 風量と24時間換気の丸めの扱いは docs/vs_simheat_example.md を参照。
input_data["ventilation_branches"] = h.ventilation["24h"]

"""##室の表面"""

input_data["surfaces"] = h.compose_surfaces(h.surface_rows["default"], surface)

"""##発熱源"""

input_data["heat_source"] = h.heat_source["default"]
input_data["humidity_source"] = h.humidity_source["default"]

"""##エアコン"""

input_data["aircon"] = h.aircon["ac4"]

"""#実行"""

from google.colab import userdata
base_url = userdata.get("VTSIMNX_API_URL")
result = vt.run_calc(base_url, input_data)
print(result.log)

series_names = [
    "vent_pressure",
    "vent_flow_rate",
    "thermal_temperature",
    "thermal_temperature_capacity",
    "thermal_temperature_layer",
    "humidity_x",
    "humidity_flux",
    "concentration_c",
    "concentration_flux",
    "thermal_heat_rate_advection",
    "thermal_heat_rate_heat_generation",
    "thermal_heat_rate_solar_gain",
    "thermal_heat_rate_nocturnal_loss",
    "thermal_heat_rate_convection",
    "thermal_heat_rate_conduction",
    "thermal_heat_rate_radiation",
    "thermal_heat_rate_capacity",
    "aircon_sensible_heat",
    "aircon_latent_heat",
    "aircon_power",
    "aircon_cop",
]
series = {name: result.get_series_df(name) for name in series_names}

df_vent_pressure                = series["vent_pressure"]
df_vent_flow_rate               = series["vent_flow_rate"]
df_thermal_temperature          = series["thermal_temperature"]
df_thermal_temperature_capacity = series["thermal_temperature_capacity"]
df_thermal_temperature_layer    = series["thermal_temperature_layer"]
df_humidity                     = series["humidity_x"]
df_humidity_flux                = series["humidity_flux"]
df_concentration                = series["concentration_c"]
df_concentration_flux           = series["concentration_flux"]
df_thermal_rate_advection       = series["thermal_heat_rate_advection"]
df_thermal_rate_heat_generation = series["thermal_heat_rate_heat_generation"]
df_thermal_rate_solar_gain      = series["thermal_heat_rate_solar_gain"]
df_thermal_rate_nocturnal_loss  = series["thermal_heat_rate_nocturnal_loss"]
df_thermal_rate_convection      = series["thermal_heat_rate_convection"]
df_thermal_rate_conduction      = series["thermal_heat_rate_conduction"]
df_thermal_rate_radiation       = series["thermal_heat_rate_radiation"]
df_thermal_rate_capacity        = series["thermal_heat_rate_capacity"]
df_ac_sensible_heat             = series["aircon_sensible_heat"]
df_ac_latent_heat               = series["aircon_latent_heat"]
df_ac_power                     = series["aircon_power"]
df_ac_cop                       = series["aircon_cop"]
