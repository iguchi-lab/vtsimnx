# -*- coding: utf-8 -*-
!pip install git+https://github.com/iguchi-lab/vtsimnx/
import vtsimnx as vt

import pandas as pd

"""#準備

##外部ファイルの読み込み
"""

df_i       = vt.read_hasp('3639999.has')                                 #通常の気象データ

"""##日射・夜間放射"""

shade_polys = {
    '和室_S': [
        [(-1.8150,  3.8000, 0.00), ( 8.6400,  3.8000, 0.00), ( 8.6400,  3.5075, 0.65), (-1.8150,  3.5075, 0.65)],   #和室（南） 南面の屋根の軒
        [(-2.7250,  0.4880, 0.00), (-1.3650,  1.1000, 0.00), (-1.3650,  1.1000, 0.45), (-2.7250,  0.4880, 0.45)],   #和室（南） 西面の屋根の軒
        [( 3.3125,  0.7800, 0.00), ( 7.1525,  0.7800, 0.00), ( 7.1525,  0.7800, 0.91), ( 3.3125,  0.7800, 0.91)],   #和室（南） バルコニー床面
        [( 3.3125,  0.7800, 0.00), ( 3.3125,  2.1500, 0.00), ( 3.3125,  2.1500, 0.91), ( 3.3125,  0.7800, 0.91)],   #和室（南） バルコニー西側壁
        [( 7.1525,  0.7800, 0.00), ( 7.1525,  2.1500, 0.00), ( 7.1525,  2.1500, 0.91), ( 7.1525,  0.7800, 0.91)],   #和室（南） バルコニー東側壁
        [( 3.3125,  0.7800, 0.91), ( 3.3125,  2.1500, 0.91), ( 7.1525,  2.1500, 0.91), ( 7.1525,  0.7800, 0.91)],   #和室（南） バルコニー前面壁
    ],
    'LD_S': [
        [(-7.0475,  3.5000, 0.00), ( 3.4075,  3.5000, 0.00), ( 3.4075,  3.2075, 0.65), (-7.0475,  3.2075, 0.65)],   #LD（南） 南面の屋根の軒
        [(-7.9575,  0.1880, 0.00), (-6.5975,  0.8000, 0.00), (-6.5975,  0.8000, 0.45), (-7.1375,  0.1880, 0.45)],   #LD（南） 西面の屋根の軒
        [(-1.9200,  0.4800, 0.00), ( 1.9200,  0.4800, 0.00), ( 1.9200,  0.4800, 0.91), (-1.9200,  0.4800, 0.91)],   #LD（南） バルコニー床面
        [(-1.9200,  0.4800, 0.00), (-1.9200,  1.8500, 0.00), (-1.9200,  1.8500, 0.91), (-1.9200,  0.4800, 0.91)],   #LD（南） バルコニー西側壁
        [( 1.9200,  0.4800, 0.00), ( 1.9200,  1.8500, 0.00), ( 1.9200,  1.8500, 0.91), ( 1.9200,  0.4800, 0.91)],   #LD（南） バルコニー東側壁
        [(-1.9200,  0.4800, 0.91), (-1.9200,  1.8500, 0.91), ( 1.9200,  1.8500, 0.91), ( 1.9200,  0.4800, 0.91)],   #LD（南） バルコニー前面壁
    ],
    '寝室_S': [
        [(-2.2700,  0.7500, 0.00), ( 8.1850,  0.7500, 0.00), ( 8.1850,  0.4575, 0.65), (-2.2700,  0.4575, 0.65)],   #寝室（南） 南面の屋根の軒
        [(-3.1800, -2.5620, 0.00), (-1.8200, -1.9500, 0.00), (-1.8200, -1.9500, 0.45), (-3.1800, -2.5620, 0.45)],   #寝室（南） 西面の屋根の軒
        [( 2.8575, -2.2700, 0.00), ( 6.6975, -2.2700, 0.00), ( 6.6975, -2.2700, 0.91), ( 2.8575, -2.2700, 0.91)],   #寝室（南） バルコニー床面
        [( 2.8575, -2.2700, 0.00), ( 2.8575, -0.9000, 0.00), ( 2.8575, -0.9000, 0.91), ( 2.8575, -2.2700, 0.91)],   #寝室（南） バルコニー西側壁
        [( 6.6975, -2.2700, 0.00), ( 6.6975, -0.9000, 0.00), ( 6.6975, -0.9000, 0.91), ( 6.6975, -2.2700, 0.91)],   #寝室（南） バルコニー東側壁
        [( 2.8575, -2.2700, 0.91), ( 2.8575, -0.9000, 0.91), ( 6.6975, -0.9000, 0.91), ( 6.6975, -2.2700, 0.91)],   #寝室（南） バルコニー前面壁
    ],
    '子供室1_S': [
        [(-6.0875,  0.7500, 0.00), ( 4.3675,  0.7500, 0.00), ( 4.3675,  0.4575, 0.65), (-6.0875,  0.4575, 0.65)],   #子供室1（南） 南面の屋根の軒
        [(-6.9975, -2.5620, 0.00), (-5.6375, -1.9500, 0.00), (-5.6375, -1.9500, 0.45), (-6.9975, -2.5620, 0.45)],   #子供室1（南） 西面の屋根の軒
        [(-0.9600, -2.2700, 0.00), ( 2.8800, -2.2700, 0.00), ( 2.8800, -2.2700, 0.91), (-0.9600, -2.2700, 0.91)],   #子供室1（南） バルコニー床面
        [(-0.9600, -2.2700, 0.00), (-0.9600, -0.9000, 0.00), (-0.9600, -0.9000, 0.91), (-0.9600, -2.2700, 0.91)],   #子供室1（南） バルコニー西側壁
        [( 2.8800, -2.2700, 0.00), ( 2.8800, -0.9000, 0.00), ( 2.8800, -0.9000, 0.91), ( 2.8800, -2.2700, 0.91)],   #子供室1（南） バルコニー東側壁
        [(-0.9600, -2.2700, 0.91), (-0.9600, -0.9000, 0.91), ( 2.8800, -0.9000, 0.91), ( 2.8800, -2.2700, 0.91)],   #子供室1（南） バルコニー前面壁
    ],
    '子供室2_S': [
        [(-8.0050,  0.7500, 0.00), ( 2.4500,  0.7500, 0.00), ( 2.4500,  0.4575, 0.65), (-8.0050,  0.4575, 0.65)],   #子供室2（南） 南面の屋根の軒
        [(-8.0950, -2.5620, 0.00), (-7.5550, -1.9500, 0.00), (-7.5550, -1.9500, 0.45), (-8.0950, -2.5620, 0.45)],   #子供室2（南） 西面の屋根の軒
        [(-2.8775, -2.2700, 0.00), ( 0.9625, -2.2700, 0.00), ( 0.9625, -2.2700, 0.91), (-2.8755, -2.2700, 0.91)],   #子供室2（南） バルコニー床面
        [(-2.8775, -2.2700, 0.00), (-2.8775, -0.9000, 0.00), (-2.8775, -0.9000, 0.91), (-2.8775, -2.2700, 0.91)],   #子供室2（南） バルコニー西側壁
        [( 0.9625, -2.2700, 0.00), ( 0.9625, -0.9000, 0.00), ( 0.9625, -0.9000, 0.91), ( 0.9625, -2.2700, 0.91)],   #子供室2（南） バルコニー東側壁
        [(-2.8775, -2.2700, 0.91), (-2.8775, -0.9000, 0.91), ( 0.9625, -0.9000, 0.91), ( 0.9625, -2.2700, 0.91)],   #子供室2（南） バルコニー前面壁
    ],
    'LD_E': [
        [(-2.4700,  5.4075, 0.00), ( 0.9100,  6.9285, 0.00), ( 0.9100,  6.9285, 0.45), (-2.4700,  5.4075, 0.45)],   #LD（東） 東面の屋根の軒（南）
        [( 0.9100,  6.9285, 0.00), ( 4.2900,  5.4075, 0.00), ( 4.2900,  5.4075, 0.45), ( 0.9100,  6.9285, 0.45)],   #LD（東） 東面の屋根の軒（北）
        [( 3.6400,  1.4190, 0.00), ( 5.6400,  0.6000, 0.00), ( 5.6400,  0.6000, 0.45), ( 3.6400,  1.4190, 0.45)],   #LD（東） 東面の屋根の軒（下屋）
    ],
    'キッチン_E': [
        [(-5.2000,  5.4075, 0.00), (-1.8200,  6.9285, 0.00), (-1.8200,  6.9285, 0.45), (-5.2000,  5.4075, 0.45)],   #キッチン（東） 東面の屋根の軒（南）
        [(-1.8200,  6.9285, 0.00), ( 1.5600,  5.4075, 0.00), ( 1.5600,  5.4075, 0.45), (-1.8200,  6.9285, 0.45)],   #キッチン（東） 東面の屋根の軒（北）
        [( 0.9100,  1.4190, 0.00), ( 2.9100,  0.6000, 0.00), ( 2.9100,  0.6000, 0.45), ( 0.9100,  1.4190, 0.45)],   #キッチン（東） 東面の屋根の軒（下屋）
    ],
    '子供室2_E': [
        [(-3.5000,  2.3075, 0.00), (-0.1200,  3.8285, 0.00), (-0.1200,  3.8285, 0.45), (-3.5000,  2.3075, 0.45)],   #子供室2（東） 東面の屋根の軒（南）
        [(-0.1200,  3.8285, 0.00), ( 3.2600,  2.3075, 0.00), ( 3.2600,  2.3075, 0.45), (-0.1200,  3.8285, 0.45)],   #子供室2（東） 東面の屋根の軒（北）
        [( 2.6100, -1.6810, 0.00), ( 4.6100, -2.5000, 0.00), ( 4.6100, -2.5000, 0.45), ( 2.6100, -1.6810, 0.45)],   #子供室2（東） 東面の屋根の軒（下屋）
    ],
    '浴室_W': [
        [(-1.6300,  0.3075, 0.00), ( 0.8400,  1.4190, 0.00), ( 0.8400,  1.4190, 0.45), (-1.6300,  0.3075, 0.45)],   #浴室（西） 北面の屋根の軒（1階）
        [( 0.1900,  3.2075, 0.00), ( 3.5700,  5.0285, 0.00), ( 3.5700,  5.0285, 3.18), ( 0.1900,  3.2075, 3.18)],   #浴室（西） 北面の屋根の軒（2階）
        [( 0.8400,  0.0000, 0.00), ( 0.8400,  3.5000, 0.00), ( 0.8400,  0.0000, 2.73), ( 0.8400,  3.5000, 2.73)],   #浴室（西） 建物外壁
        [( 0.3900,  1.1000, 2.73), ( 6.7500,  1.1000, 2.73), ( 6.7500,  1.1000, 4.29), ( 0.3900,  0.4025, 4.29)],   #浴室（西） 西面の屋根の軒（1階）
    ],
    'クローゼット_W': [
        [(-1.5600,  0.1075, 0.00), ( 1.8200,  1.9285, 0.00), ( 1.8200,  1.9285, 0.45), (-1.5600,  0.1075, 0.45)],   #クローゼット（西） 西面の屋根の軒（北）
        [(1.82000,  1.9285, 0.00), ( 5.2000,  0.1075, 0.00), ( 5.2000,  0.1075, 0.45), ( 1.8200,  1.9285, 0.45)],   #クローゼット（西） 西面の屋根の軒（南）
    ],
    '寝室_W': [
        [(-4.2900,  0.1075, 0.00), (-0.9100,  1.9285, 0.00), (-0.9100,  1.9285, 0.45), (-4.2900,  0.1075, 0.45)],   #寝室 西面の屋根の軒（北）
        [(-0.9100,  1.9285, 0.00), ( 2.4700,  0.1075, 0.00), ( 2.4700,  0.1075, 0.45), (-0.9100,  1.9285, 0.45)],   #寝室 西面の屋根の軒（南）
    ],
    '1階トイレ_N': [
        [(-3.1800,  0.6000, 0.00), ( 4.5450,  0.6000, 0.00), ( 4.5450,  0.3075, 0.65), (-3.1800,  0.3075, 0.65)],   #1階トイレ（北） 北面の屋根の軒（1階）
    ],
    '洗面所_N': [
        [(-5.0000,  0.3000, 0.00), ( 2.7250,  0.3000, 0.00), ( 2.7250,  0.0075, 0.65), (-5.0000,  0.0075, 0.65)],   #洗面所（北） 北面の屋根の軒(1階)
    ],
    '1階ホール_N': [
        [( 0.4550,  0.3075, 2.47), ( 0.0000,  0.3075, 2.47), ( 0.0000,  1.4190, 0.00), ( 0.4550,  1.4190, 0.00)],   #1階ホール（北） 北面の屋根の軒（1階）
        [(-0.4550,  3.5000, 0.00), ( 2.7250,  3.5000, 0.00), ( 2.7250,  3.2075, 0.65), (-0.4550,  3.2075, 0.65)],   #1階ホール（北） 北面の屋根の軒（2階）
        [( 0.4550,  0.6000, 1.82), ( 0.4550,  0.0000, 1.82), ( 0.4550,  0.0000, 0.00), ( 0.4550,  0.8190, 0.00)],   #1階ホール（北） 建物外壁
        [( 2.2750,  1.1000, 0.00), ( 3.8350,  0.4025, 0.00), ( 3.8350,  0.4025, 0.45), ( 2.2750,  1.1000, 0.45)],   #1階ホール（北） 西面の屋根の軒（1階）
    ],
    '2階トイレ_N': [
        [(-1.8150,  0.9000, 0.00), ( 8.6400,  0.9000, 0.00), ( 8.6400,  0.6075, 0.65), (-1.8150,  0.6075, 0.65)],   #2階トイレ（北） 北面の屋根の軒（2階）
    ],
    '2階ホール_N': [
        [(-3.6350,  0.7000, 0.00), ( 6.8200,  0.7000, 0.00), ( 6.8200,  0.4075, 0.65), (-3.6350,  0.4075, 0.65)],   #2階ホール（北） 北面の屋根の軒(2階)
    ]
}

solar     = pd.DataFrame(index=df_i.index)
nocturnal = pd.DataFrame(index=df_i.index)

dni           = df_i["直達日射量"]
dhi           = df_i["水平面拡散日射量"]
rn_horizontal = df_i["夜間放射量"]
lat_deg       = 36.0    #35.69
lon_deg       = 140.0   #139.77

# 壁（鉛直）
solar["日射熱取得量（南面）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=0.0,  tilt_deg=90.0)
solar["日射熱取得量（南面ガラス）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=0.0,  tilt_deg=90.0, glass=True)
solar["日射熱取得量（西面）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=90.0,  tilt_deg=90.0)
solar["日射熱取得量（西面ガラス）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=90.0,  tilt_deg=90.0, glass=True)
solar["日射熱取得量（東面）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=-90.0, tilt_deg=90.0)
solar["日射熱取得量（東面ガラス）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=-90.0, tilt_deg=90.0, glass=True)
solar["日射熱取得量（北面）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=180.0, tilt_deg=90.0)
solar["日射熱取得量（北面ガラス）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=180.0, tilt_deg=90.0, glass=True)

# 水平
solar["日射熱取得量（水平上向き）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=0.0, tilt_deg=0.0)
solar["日射熱取得量（水平下向き）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=0.0, tilt_deg=180.0)

# 屋根（傾斜）
solar["日射熱取得量（南面屋根）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=0.0, tilt_deg=24.2277)
solar["日射熱取得量（北面屋根）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=180.0, tilt_deg=24.2277)
solar["日射熱取得量（北面屋根2）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=180.0, tilt_deg=19.29)
solar["日射熱取得量（西面屋根）"] =\
    vt.solar_gain_by_angles(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg, azimuth_deg=90.0, tilt_deg=25.5651)

#夜間放射量
nocturnal['夜間放射量_水平'] =\
    vt.nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=0.0)
nocturnal['夜間放射量_垂直'] =\
    vt.nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=90.0)
nocturnal['夜間放射量_切妻'] =\
    vt.nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=24.2277)
nocturnal['夜間放射量_北面屋根2'] =\
    vt.nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=19.29)
nocturnal['夜間放射量_西面屋根'] =\
    vt.nocturnal_gain_by_angles(rn_horizontal=rn_horizontal, tilt_deg=25.5651)

#庇等あり
solar["日射熱取得量（和室_南面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=0.0, tilt_deg=90.0, window_width=2.250, window_height=1.800,
                                       shade_coords=shade_polys['和室_S'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（LD_南面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=0.0, tilt_deg=90.0, window_width=3.300, window_height=2.100,
                                       shade_coords=shade_polys['LD_S'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（寝室_南面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=0.0, tilt_deg=90.0, window_width=1.650, window_height=1.050,
                                       shade_coords=shade_polys['寝室_S'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（子供室1_南面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=0.0, tilt_deg=90.0, window_width=1.650, window_height=1.950,
                                       shade_coords=shade_polys['子供室1_S'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（子供室2_南面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=0.0, tilt_deg=90.0, window_width=1.650, window_height=1.950,
                                       shade_coords=shade_polys['子供室2_S'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（LD_東面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=-90.0, tilt_deg=90.0, window_width=1.650, window_height=1.300,
                                       shade_coords=shade_polys['LD_E'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（キッチン_東面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=-90.0, tilt_deg=90.0, window_width=1.400, window_height=0.700,
                                       shade_coords=shade_polys['キッチン_E'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（子供室2_東面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=-90.0, tilt_deg=90.0, window_width=0.600, window_height=1.100,
                                       shade_coords=shade_polys['子供室2_E'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（浴室_西面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=90.0, tilt_deg=90.0, window_width=0.600, window_height=0.900,
                                       shade_coords=shade_polys['浴室_W'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（クローゼット_西面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=90.0, tilt_deg=90.0, window_width=0.600, window_height=0.900,
                                       shade_coords=shade_polys['クローゼット_W'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（寝室_西面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=90.0, tilt_deg=90.0, window_width=0.900, window_height=1.100,
                                       shade_coords=shade_polys['寝室_W'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（1階トイレ_北面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=180.0, tilt_deg=90.0, window_width=0.600, window_height=0.900,
                                       shade_coords=shade_polys['1階トイレ_N'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（洗面所_北面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=180.0, tilt_deg=90.0, window_width=0.600, window_height=0.900,
                                       shade_coords=shade_polys['洗面所_N'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（1階ホール_北面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=180.0, tilt_deg=90.0, window_width=0.900, window_height=0.900,
                                       shade_coords=shade_polys['1階ホール_N'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（2階トイレ_北面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=180.0, tilt_deg=90.0, window_width=0.600, window_height=0.900,
                                       shade_coords=shade_polys['2階トイレ_N'], shade_origin_mode="window_top_center", glass=True)

solar["日射熱取得量（2階ホール_北面ガラス）"] =\
    vt.solar_gain_by_angles_with_shade(dni=dni, dhi=dhi, lat_deg=lat_deg, lon_deg=lon_deg,
                                       azimuth_deg=180.0, tilt_deg=90.0, window_width=0.900, window_height=1.100,
                                       shade_coords=shade_polys['2階ホール_N'], shade_origin_mode="window_top_center", glass=True)

"""#地盤温度"""

df_i['地下1m温度'] =\
    vt.ground_temperature_by_depth(
        depth_m                         = 1.0,
        t_out                           = df_i['外気温'],
        solar_horizontal                = df_i['水平面拡散日射量'] + df_i['直達日射量'],
        nocturnal_horizontal            = df_i['夜間放射量'],
        deep_layer_depth_m              = 10.0,
        deep_layer_temp_c               = 10.0,
        thermal_conductivity_w_mk       = 1.5,
        volumetric_heat_capacity_j_m3k  = 2.2e6,
        solar_to_surface_temp_coeff     = 0.003,
        nocturnal_to_surface_temp_coeff = 0.003,
        spinup                          = True
    )

"""##室の気積"""

room_volume = {
    '床下':         65.42 * 0.5,  #実際の気積は13.08m3
    '和室':         39.75,
    'LD':           51.67,
    '台所':         19.25,
    '浴室':          7.45,
    '1階トイレ':     3.73,
    '洗面所':        7.45,
    'ホール':       32.30,
    'クローゼット': 11.92,
    '寝室':         31.80,
    '子供室1':      25.83,
    '子供室2':      25.84,
    '2階ホール':    27.29,
    '切妻':         52.18 * 0.9,  #実際の気積は47.70m3
    '片流れ(3)':    12.42 * 0.9,  #実際の気積は 5.82m3
    '片流れ(4)':     3.31 * 0.9,  #実際の気積は 1.13m3
    '階間(L4)':     24.64,
    '2階トイレ':     3.97
}

"""##層の設定"""

materials  = vt.materials

layers = {
    '地盤': [
        {'key': 'コンクリート',                     **materials['コンクリート'],                    't': 0.120},
        {'key': '地盤',                            'lambda':   0.698,  'v_capa':  3000*1000,        't': 1.00}
    ],
    '基礎外壁': [
        {'key': 'コンクリート',                     **materials['コンクリート'],                    't': 0.120}
    ],
    'ホール-床下': [
        {'key': 'コンクリート',                     **materials['コンクリート'],                    't': 0.120},
        {'key': '押出法ポリスチレンフォーム3種',    **materials['押出法ポリスチレンフォーム3種'],   't': 0.022}
    ],
    '外壁_一般部': [
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                  't': 0.0095},
        {'key': '中空層',                           'air_layer': True, 'thermal_resistance': 0.09,  't': 0.024},  #中空層
        {'key': '住宅用グラスウール断熱材16K相当',  **materials['住宅用グラスウール断熱材16K相当'], 't': 0.076},
        {'key': '合板',                             **materials['合板'],                            't': 0.012},
        {'key': '通気層',                           'ventilated_air_layer': True,                   't': 0.018},  #通気層
        {'key': '木片セメント板',                   **materials['木片セメント板'],                  't': 0.015}
    ],
    '外壁_熱橋部': [
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                  't': 0.0095},
        {'key': '天然木材1類(桧、杉、えぞ松等)',    **materials['天然木材1類(桧、杉、えぞ松等)'],   't': 0.100},
        {'key': '合板',                             **materials['合板'],                            't': 0.012},
        {'key': '通気層',                           'ventilated_air_layer': True,                   't': 0.018},  #通気層
        {'key': '木片セメント板',                   **materials['木片セメント板'],                  't': 0.015}
    ],
    '外皮床_一般部': [
        {'key': '合板',                             **materials['合板'],                            't': 0.012},
        {'key': '住宅用グラスウール断熱材16K相当',  **materials['住宅用グラスウール断熱材16K相当'], 't': 0.082}
    ],
    '外皮床_熱橋部': [
        {'key': '合板',                             **materials['合板'],                            't': 0.012},
        {'key': '天然木材1類(桧、杉、えぞ松等)',    **materials['天然木材1類(桧、杉、えぞ松等)'],   't': 0.082}
    ],
    '外皮天井': [
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                   't': 0.0095},

        {'key': '住宅用グラスウール断熱材10K相当',  **materials['住宅用グラスウール断熱材10K相当'],  't': 0.171},
    ],
    '間仕切壁': [
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                   't': 0.0125},
        {'key': '中空層',                           'air_layer': True, 'thermal_resistance': 0.09,   't': 0.1000}, #中空層
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                   't': 0.0125}
    ],
    '床': [
        {'key': '合板',                             **materials['合板'],                            't': 0.012}
    ],
    '天井': [
        {'key': 'せっこうボード',                   **materials['せっこうボード'],                  't': 0.0125}
    ],
    '屋根': [
        {'key': '合板',                             **materials['合板'],                            't': 0.012}
    ]
}

"""##表面の設定"""

u_value_window = 12.0
surface = {
    '地盤':         {'part': 'floor', 'layers': layers['地盤']},

    'E_基礎外壁': {'part': 'wall', 'layers': layers['基礎外壁'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（東面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_基礎外壁': {'part': 'wall', 'layers': layers['基礎外壁'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_基礎外壁': {'part': 'wall', 'layers': layers['基礎外壁'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（南面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'W_基礎外壁': {'part': 'wall', 'layers': layers['基礎外壁'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（西面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},

    'ホール-床下': {'part': 'wall', 'layers': layers['ホール-床下'], 'alpha_i': 2.0, 'alpha_o': 2.0 + 4.7},

    '0_外壁_一般部': {'part': 'wall', 'layers': layers['外壁_一般部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7},
    'E_外壁_一般部': {'part': 'wall', 'layers': layers['外壁_一般部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（東面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_外壁_一般部': {'part': 'wall', 'layers': layers['外壁_一般部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_外壁_一般部': {'part': 'wall', 'layers': layers['外壁_一般部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（南面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'W_外壁_一般部': {'part': 'wall', 'layers': layers['外壁_一般部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（西面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},

    '0_外壁_熱橋部': {'part': 'wall', 'layers': layers['外壁_熱橋部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7},
    'E_外壁_熱橋部': {'part': 'wall', 'layers': layers['外壁_熱橋部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（東面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_外壁_熱橋部': {'part': 'wall', 'layers': layers['外壁_熱橋部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_外壁_熱橋部': {'part': 'wall', 'layers': layers['外壁_熱橋部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（南面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'W_外壁_熱橋部': {'part': 'wall', 'layers': layers['外壁_熱橋部'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（西面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},

    '外皮床_一般部': {'part': 'floor', 'layers': layers['外皮床_一般部'], 'alpha_i': 2.0, 'alpha_o': 2.0 + 4.7},
    '外皮床_熱橋部': {'part': 'floor', 'layers': layers['外皮床_熱橋部'], 'alpha_i': 2.0, 'alpha_o': 2.0 + 4.7},
    '外皮天井':      {'part': 'floor', 'layers': layers['外皮床_一般部'], 'alpha_i': 6.4, 'alpha_o': 6.4 + 4.7},

    '南面屋根':      {'part': 'ceiling', 'layers': layers['屋根'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（南面屋根）'], 'nocturnal': nocturnal['夜間放射量_切妻']},
    '北面屋根':      {'part': 'ceiling', 'layers': layers['屋根'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面屋根）'], 'nocturnal': nocturnal['夜間放射量_切妻']},
    '北面屋根2':     {'part': 'ceiling', 'layers': layers['屋根'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面屋根2）'], 'nocturnal': nocturnal['夜間放射量_北面屋根2']},
    '西面屋根':      {'part': 'ceiling', 'layers': layers['屋根'], 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（西面屋根）'], 'nocturnal': nocturnal['夜間放射量_西面屋根']},

    '間仕切壁':      {'part': 'wall',    'layers': layers['間仕切壁'], 'alpha_i': 4.4, 'alpha_o': 4.4},
    '床':            {'part': 'floor',   'layers': layers['床'],       'alpha_i': 2.0, 'alpha_o': 2.0},
    '天井':          {'part': 'floor',   'layers': layers['天井'],     'alpha_i': 6.4, 'alpha_o': 6.4},

    '室内建具':      {'part': 'wall', 'u_value': 2.33, 'alpha_i': 4.4, 'alpha_o': 4.4},
    'W_玄関ドア':    {'part': 'wall', 'u_value': 2.33, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（西面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_台所ドア':    {'part': 'wall', 'u_value': 2.33, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（北面）'], 'nocturnal': nocturnal['夜間放射量_垂直']},

    'S_和室_窓':         {'part': 'glass', 'u_value': 1 / (1 / u_value_window + 0.069), 'SCR': 0.342, 'SCC': 0.087, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（和室_南面ガラス）'],         'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_LD_窓':           {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（LD_南面ガラス）'],           'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_寝室_窓':         {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（寝室_南面ガラス）'],         'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_子供室1_窓':      {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（子供室1_南面ガラス）'],      'nocturnal': nocturnal['夜間放射量_垂直']},
    'S_子供室2_窓':      {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（子供室2_南面ガラス）'],      'nocturnal': nocturnal['夜間放射量_垂直']},

    'E_LD_窓':           {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（LD_東面ガラス）'],           'nocturnal': nocturnal['夜間放射量_垂直']},
    'E_キッチン_窓':     {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（キッチン_東面ガラス）'],     'nocturnal': nocturnal['夜間放射量_垂直']},
    'E_子供室2_窓':      {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（子供室2_東面ガラス）'],      'nocturnal': nocturnal['夜間放射量_垂直']},

    'W_浴室_窓':         {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（浴室_西面ガラス）'],         'nocturnal': nocturnal['夜間放射量_垂直']},
    'W_クローゼット_窓': {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（クローゼット_西面ガラス）'], 'nocturnal': nocturnal['夜間放射量_垂直']},
    'W_寝室_窓':         {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.491, 'SCC': 0.139, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（寝室_西面ガラス）'],         'nocturnal': nocturnal['夜間放射量_垂直']},

    'N_1階トイレ_窓':    {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（1階トイレ_北面ガラス）'],    'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_洗面所_窓':       {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（洗面所_北面ガラス）'],       'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_1階ホール_窓':    {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（1階ホール_北面ガラス）'],    'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_2階トイレ_窓':    {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（2階トイレ_北面ガラス）'],    'nocturnal': nocturnal['夜間放射量_垂直']},
    'N_2階ホール_窓':    {'part': 'glass', 'u_value': u_value_window, 'SCR': 0.986, 'SCC': 0.014, 'alpha_i': 4.4, 'alpha_o': 20.3 + 4.7, 'solar': solar['日射熱取得量（2階ホール_北面ガラス）'],    'nocturnal': nocturnal['夜間放射量_垂直']},
}

"""#input_dataの作成"""

input_data = {}

"""##シミュレーションの設定"""

input_data['simulation'] = {
    'index': df_i.index,
    'tolerance': {
        'ventilation': 1e-6, 'thermal': 1e-6, 'convergence': 1e-6
    }
}

"""##ノード"""

t_capa_per_m3 = 12.6 * 1000
m_capa_per_m3 = 25.1 * 1000 * 1000

input_data['nodes'] = [
    {'key': '外部',         't': df_i['外気温'], 'x': df_i['外気絶対湿度'] / 1000},
    {'key': '地下1m',       't': df_i['地下1m温度']},
    {'key': '床下',         'calc_t': True, 'calc_x': True, 'v': room_volume['床下'],
                            'thermal_mass': room_volume['床下']              * t_capa_per_m3,  #J/K
                            'moisture_capacity': room_volume['床下']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},                           #J/(kg/kg')
    {'key': '和室',         'calc_t': True, 'calc_x': True, 'v': room_volume['和室'],
                            'thermal_mass': room_volume['和室']              * t_capa_per_m3,
                            'moisture_capacity': room_volume['和室']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': 'LD',           'calc_t': True, 'calc_x': True, 'v': room_volume['LD'],
                            'thermal_mass': room_volume['LD']                * t_capa_per_m3,
                            'moisture_capacity': room_volume['LD']           * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '台所',         'calc_t': True, 'calc_x': True, 'v': room_volume['台所'],
                            'thermal_mass': room_volume['台所']              * t_capa_per_m3,
                            'moisture_capacity': room_volume['台所']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '浴室',         'calc_t': True, 'calc_x': True, 'v': room_volume['浴室'],
                            'thermal_mass': room_volume['浴室']              * t_capa_per_m3,
                            'moisture_capacity': room_volume['浴室']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '1階トイレ',    'calc_t': True, 'calc_x': True, 'v': room_volume['1階トイレ'],
                            'thermal_mass': room_volume['1階トイレ']         * t_capa_per_m3,
                            'moisture_capacity': room_volume['1階トイレ']    * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '洗面所',       'calc_t': True, 'calc_x': True, 'v': room_volume['洗面所'],
                            'thermal_mass': room_volume['洗面所']            * t_capa_per_m3,
                            'moisture_capacity': room_volume['洗面所']       * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': 'ホール',       'calc_t': True, 'calc_x': True, 'v': room_volume['ホール'],
                            'thermal_mass': room_volume['ホール']            * t_capa_per_m3,
                            'moisture_capacity': room_volume['ホール']       * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': 'クローゼット', 'calc_t': True, 'calc_x': True, 'v': room_volume['クローゼット'],
                            'thermal_mass': room_volume['クローゼット']      * t_capa_per_m3,
                            'moisture_capacity': room_volume['クローゼット'] * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '寝室',         'calc_t': True, 'calc_x': True, 'v': room_volume['寝室'],
                            'thermal_mass': room_volume['寝室']              * t_capa_per_m3,
                            'moisture_capacity': room_volume['寝室']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '子供室1',      'calc_t': True, 'calc_x': True, 'v': room_volume['子供室1'],
                            'thermal_mass': room_volume['子供室1']           * t_capa_per_m3,
                            'moisture_capacity': room_volume['子供室1']      * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '子供室2',      'calc_t': True, 'calc_x': True, 'v': room_volume['子供室2'],
                            'thermal_mass': room_volume['子供室2']           * t_capa_per_m3,
                            'moisture_capacity': room_volume['子供室2']      * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '2階ホール',    'calc_t': True, 'calc_x': True, 'v': room_volume['2階ホール'],
                            'thermal_mass': room_volume['2階ホール']         * t_capa_per_m3,
                            'moisture_capacity': room_volume['2階ホール']    * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '切妻',         'calc_t': True, 'calc_x': True, 'v': room_volume['切妻'],
                            'thermal_mass': room_volume['切妻']              * t_capa_per_m3,
                            'moisture_capacity': room_volume['切妻']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '片流れ(3)',    'calc_t': True, 'calc_x': True, 'v': room_volume['片流れ(3)'],
                            'thermal_mass': room_volume['片流れ(3)']         * t_capa_per_m3,
                            'moisture_capacity': room_volume['片流れ(3)']    * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '片流れ(4)',    'calc_t': True, 'calc_x': True, 'v': room_volume['片流れ(4)'],
                            'thermal_mass': room_volume['片流れ(4)']         * t_capa_per_m3,
                            'moisture_capacity': room_volume['片流れ(4)']    * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '階間(L4)',     'calc_t': True, 'calc_x': True, 'v': room_volume['階間(L4)'],
                            'thermal_mass': room_volume['階間(L4)']          * t_capa_per_m3,
                            'moisture_capacity': room_volume['和室']         * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"},
    {'key': '2階トイレ',    'calc_t': True, 'calc_x': True, 'v': room_volume['2階トイレ'],
                            'thermal_mass': room_volume['2階トイレ']         * t_capa_per_m3,
                            'moisture_capacity': room_volume['2階トイレ']    * m_capa_per_m3,
                            "moisture_capacity_unit": "J/(kg/kg')"}
]

"""##換気回路網"""

input_data['ventilation_branches'] = [
    {'key': '外部->和室->ホール',                  'vol':   20     / 3600},                         #24時間換気
    {'key': '外部->台所->LD->ホール',              'vol':   20     / 3600},                         #24時間換気
    {'key': '外部->LD->ホール',                    'vol':   40     / 3600},                         #24時間換気
    {'key': 'ホール->外部',                        'vol':   80 / 3 / 3600},                         #24時間換気
    {'key': 'ホール->洗面所->外部',                'vol':   80 / 3 / 3600},                         #24時間換気
    {'key': 'ホール->1階トイレ->外部',             'vol':   80 / 3 / 3600},                         #24時間換気
    {'key': '外部->クローゼット->寝室->2階ホール', 'vol':   20     / 3600},                         #24時間換気
    {'key': '外部->寝室->2階ホール',               'vol':   20     / 3600},                         #24時間換気
    {'key': '外部->子供室1->2階ホール',            'vol':   20     / 3600},                         #24時間換気
    {'key': '外部->子供室2->2階ホール',            'vol':   20     / 3600},                         #24時間換気
    {'key': '2階ホール->2階トイレ->外部',          'vol':   40     / 3600},                         #24時間換気
    {'key': '2階ホール->外部',                     'vol':   40     / 3600},                         #24時間換気
    {'key': 'ホール->2階ホール->ホール',           'vol':  273     / 3600},                         #室間相互換気
    {'key': 'LD->台所->LD',                        'vol': 5000     / 3600},                         #室間相互換気
    {'key': '外部->台所->外部',                    'vol': vt.schedule.vol['LD']},                   #局所排気
    {'key': '外部->ホール->洗面所->浴室->外部',    'vol': vt.schedule.vol['浴室']},                 #局所排気
    {'key': '外部->ホール->1階トイレ->外部',       'vol': vt.schedule.vol['1階トイレ']},            #局所排気
    {'key': '外部->床下->外部',                    'vol': room_volume['床下'] * 5.0 / 3600},        #床下換気
    {'key': '外部->切妻->外部',                    'vol': room_volume['切妻'] * 5.0 / 3600},        #小屋裏換気
    {'key': '外部->片流れ(3)->外部',               'vol': room_volume['片流れ(3)'] * 5.0 / 3600},   #小屋裏換気
    {'key': '外部->片流れ(4)->外部',               'vol': room_volume['片流れ(4)'] * 5.0 / 3600},   #小屋裏換気
]

"""##室の表面"""

input_data['surfaces'] = [
    {'key': '床下->外部||W_外壁',                  **surface['W_基礎外壁'],        'area':  0.93 + 1.86},
    {'key': '床下->外部||E_外壁',                  **surface['E_基礎外壁'],        'area':  3.71},
    {'key': '床下->外部||S_外壁',                  **surface['S_基礎外壁'],        'area':  5.34},
    {'key': '床下->外部||N_外壁',                  **surface['N_基礎外壁'],        'area':  0.7 + 0.46 + 3.48},
    {'key': '床下->地下1m',                        **surface['地盤'],              'area':  65.42},

    {'key': '和室->外部||W_外壁_一般部',           **surface['W_外壁_一般部'],     'area':  8.74 * 0.83},
    {'key': '和室->外部||W_外壁_熱橋部',           **surface['W_外壁_熱橋部'],     'area':  8.74 * 0.17},
    {'key': '和室->外部||S_外壁_一般部',           **surface['S_外壁_一般部'],     'area': (10.92 - 4.59) * 0.83},
    {'key': '和室->外部||S_外壁_熱橋部',           **surface['S_外壁_熱橋部'],     'area': (10.92 - 4.59) * 0.17},
    {'key': '和室->外部||S_窓_障子',               **surface['S_和室_窓'],         'area':  4.59},
    {'key': '和室->外部||N_外壁_一般部',           **surface['N_外壁_一般部'],     'area':  2.18 * 0.83},
    {'key': '和室->外部||N_外壁_熱橋部',           **surface['N_外壁_熱橋部'],     'area':  2.18 * 0.17},
    {'key': '和室->床下||床_一般部',               **surface['外皮床_一般部'],     'area': 16.56 * 0.80},
    {'key': '和室->床下||床_熱橋部',               **surface['外皮床_熱橋部'],     'area': 16.56 * 0.20},

    {'key': 'LD->外部||E_外壁_一般部',             **surface['E_外壁_一般部'],     'area': (8.74 - 2.145) * 0.83},
    {'key': 'LD->外部||E_外壁_熱橋部',             **surface['E_外壁_熱橋部'],     'area': (8.74 - 2.145) * 0.17},
    {'key': 'LD->外部||E_窓_レース',               **surface['E_LD_窓'],           'area': 2.145},
    {'key': 'LD->外部||S_外壁_一般部',             **surface['S_外壁_一般部'],     'area': (14.20 - 3.465 - 3.465) * 0.83},
    {'key': 'LD->外部||S_外壁_熱橋部',             **surface['S_外壁_熱橋部'],     'area': (14.20 - 3.465 - 3.465) * 0.17},
    {'key': 'LD->外部||S_窓_レース',               **surface['S_LD_窓'],           'area': 3.465 + 3.465},
    {'key': 'LD->床下||外皮床_一般部',             **surface['外皮床_一般部'],     'area': 21.53 * 0.80},
    {'key': 'LD->床下||外皮床_熱橋部',             **surface['外皮床_熱橋部'],     'area': 21.53 * 0.20},
    {'key': 'LD->和室||間仕切壁',                  **surface['間仕切壁'],          'area':  8.74 - 3.06},
    {'key': 'LD->和室||建具',                      **surface['室内建具'],          'area':  3.06},

    {'key': '台所->外部||E_外壁_一般部',           **surface['E_外壁_一般部'],     'area': (8.46 - 0.98) * 0.83},
    {'key': '台所->外部||E_外壁_熱橋部',           **surface['E_外壁_熱橋部'],     'area': (8.46 - 0.98) * 0.17},
    {'key': '台所->外部||E_窓',                    **surface['E_キッチン_窓'],     'area':  0.98},
    {'key': '台所->外部||N_外壁_一般部',           **surface['N_外壁_一般部'],     'area': (5.12 - 1.62) * 0.83},
    {'key': '台所->外部||N_外壁_熱橋部',           **surface['N_外壁_熱橋部'],     'area': (5.12 - 1.62) * 0.17},
    {'key': '台所->外部||N_台所ドア',              **surface['N_台所ドア'],        'area':  1.62},
    {'key': '台所->LD||間仕切壁',                  **surface['間仕切壁'],          'area':  5.46 - 2.0},
    {'key': '台所->床下||外皮床_一般部',           **surface['外皮床_一般部'],     'area':  8.28 * 0.80},
    {'key': '台所->床下||外皮床_熱橋部',           **surface['外皮床_熱橋部'],     'area':  8.28 * 0.20},

    {'key': '浴室->外部||W_外壁_一般部',           **surface['W_外壁_一般部'],     'area':  (4.09 - 0.54) * 0.83},
    {'key': '浴室->外部||W_外壁_熱橋部',           **surface['W_外壁_熱橋部'],     'area':  (4.09 - 0.54) * 0.17},
    {'key': '浴室->外部||W_窓',                    **surface['W_浴室_窓'],         'area':  0.54},
    {'key': '浴室->外部||N_外壁_一般部',           **surface['N_外壁_一般部'],     'area':  4.09 * 0.83},
    {'key': '浴室->外部||N_外壁_熱橋部',           **surface['N_外壁_熱橋部'],     'area':  4.09 * 0.17},
    {'key': '浴室->床下||外皮床_一般部',           **surface['外皮床_一般部'],     'area':  3.31 * 0.80},
    {'key': '浴室->床下||外皮床_熱橋部',           **surface['外皮床_熱橋部'],     'area':  3.31 * 0.20},

    {'key': '1階トイレ->外部||N_外壁_一般部',      **surface['N_外壁_一般部'],     'area':  (2.05 - 0.54) * 0.83},
    {'key': '1階トイレ->外部||N_外壁_熱橋部',      **surface['N_外壁_熱橋部'],     'area':  (2.05 - 0.54) * 0.17},
    {'key': '1階トイレ->外部||N_窓',               **surface['N_1階トイレ_窓'],    'area':  0.54},
    {'key': '1階トイレ->台所||間仕切壁',           **surface['間仕切壁'],          'area':  4.09},
    {'key': '1階トイレ->床下||外皮床_一般部',      **surface['外皮床_一般部'],     'area':  1.66 * 0.80},
    {'key': '1階トイレ->床下||外皮床_熱橋部',      **surface['外皮床_熱橋部'],     'area':  1.66 * 0.20},

    {'key': '洗面所->外部||N_外壁_一般部',         **surface['N_外壁_一般部'],     'area':  (4.09 - 0.54) * 0.83},
    {'key': '洗面所->外部||N_外壁_熱橋部',         **surface['N_外壁_熱橋部'],     'area':  (4.09 - 0.54) * 0.17},
    {'key': '洗面所->外部||N_窓',                  **surface['N_洗面所_窓'],       'area':  0.54},
    {'key': '洗面所->浴室||間仕切壁',              **surface['間仕切壁'],          'area':  4.09 - 1.422},
    {'key': '洗面所->浴室||室内建具',              **surface['室内建具'],          'area':  1.422},
    {'key': '洗面所->1階トイレ||間仕切壁',         **surface['間仕切壁'],          'area':  4.09},
    {'key': '洗面所->床下||外皮床_一般部',         **surface['外皮床_一般部'],     'area':  3.31 * 0.80},
    {'key': '洗面所->床下||外皮床_熱橋部',         **surface['外皮床_熱橋部'],     'area':  3.31 * 0.20},

    {'key': 'ホール->外部||W_外壁_一般部',         **surface['W_外壁_一般部'],     'area':  (5.3 - 1.89) * 0.83},
    {'key': 'ホール->外部||W_外壁_熱橋部',         **surface['W_外壁_熱橋部'],     'area':  (5.3 - 1.89) * 0.17},
    {'key': 'ホール->外部||W_玄関ドア',            **surface['W_玄関ドア'],        'area':  1.89},
    {'key': 'ホール->外部||N_外壁_一般部',         **surface['N_外壁_一般部'],     'area':  (3.97 + 3.28 - 0.54) * 0.83},   #simheatとの差を確認
    {'key': 'ホール->外部||N_外壁_熱橋部',         **surface['N_外壁_熱橋部'],     'area':  (3.97 + 3.28 - 0.54) * 0.17},
    {'key': 'ホール->外部||N_窓',                  **surface['N_1階ホール_窓'],    'area':  0.54},
    {'key': 'ホール->地下1m',                      **surface['地盤'],              'area':  2.40},
    {'key': 'ホール->和室||間仕切壁',              **surface['間仕切壁'],          'area':  (5.46 + 3.28) - 1.422},
    {'key': 'ホール->和室||建具',                  **surface['室内建具'],          'area':  1.422},
    {'key': 'ホール->LD||間仕切壁',                **surface['間仕切壁'],          'area':  8.74 - 1.422},
    {'key': 'ホール->LD||室内建具',                **surface['室内建具'],          'area':  1.422},
    {'key': 'ホール->台所||間仕切壁',              **surface['間仕切壁'],          'area':  4.37 - 1.422},
    {'key': 'ホール->台所||室内建具',              **surface['室内建具'],          'area':  1.422},
    {'key': 'ホール->浴室||間仕切壁',              **surface['間仕切壁'],          'area':  4.09},
    {'key': 'ホール->1階トイレ||間仕切壁',         **surface['間仕切壁'],          'area':  2.05 - 1.422},
    {'key': 'ホール->1階トイレ||室内建具',         **surface['室内建具'],          'area':  1.422},
    {'key': 'ホール->洗面所||間仕切壁',            **surface['間仕切壁'],          'area':  4.09 - 1.422},
    {'key': 'ホール->洗面所||室内建具',            **surface['室内建具'],          'area':  1.422},
    {'key': 'ホール->床下||外皮床_一般部',         **surface['外皮床_一般部'],     'area': 10.77 * 0.80},
    {'key': 'ホール->床下||外皮床_熱橋部',         **surface['外皮床_熱橋部'],     'area': 10.77 * 0.20},
    {'key': 'ホール->床下||ホール-床下',           **surface['ホール-床下'],       'area':  0.70 + 0.93},

    {'key': 'クローゼット->外部||W_外壁_一般部',   **surface['W_外壁_一般部'],     'area':  (4.37 - 0.54) * 0.83},
    {'key': 'クローゼット->外部||W_外壁_熱橋部',   **surface['W_外壁_熱橋部'],     'area':  (4.37 - 0.54) * 0.17},
    {'key': 'クローゼット->外部||W_窓',            **surface['W_クローゼット_窓'], 'area':  0.54},
    {'key': 'クローゼット->外部||N_外壁_一般部',   **surface['N_外壁_一般部'],     'area':  6.55 * 0.83},
    {'key': 'クローゼット->外部||N_外壁_熱橋部',   **surface['N_外壁_熱橋部'],     'area':  6.55 * 0.17},
    {'key': 'クローゼット->階間(L4)||床',          **surface['床'],                'area':  4.97},

    {'key': '寝室->外部||S_外壁_一般部',           **surface['S_外壁_一般部'],     'area':  (8.74 - 1.733) * 0.83},
    {'key': '寝室->外部||S_外壁_熱橋部',           **surface['S_外壁_熱橋部'],     'area':  (8.74 - 1.733) * 0.17},
    {'key': '寝室->外部||S_窓_レース',             **surface['S_寝室_窓'],         'area':  1.733},
    {'key': '寝室->外部||W_外壁_一般部',           **surface['W_外壁_一般部'],     'area':  (8.74 - 0.99) * 0.83},
    {'key': '寝室->外部||W_外壁_熱橋部',           **surface['W_外壁_熱橋部'],     'area':  (8.74 - 0.99) * 0.17},
    {'key': '寝室->外部||W_窓_レース',             **surface['W_寝室_窓'],         'area':  0.99},
    {'key': '寝室->クローゼット||間仕切壁',        **surface['間仕切壁'],          'area':  6.55 - 1.422},
    {'key': '寝室->クローゼット||室内建具',        **surface['室内建具'],          'area':  1.422},
    {'key': '寝室->階間(L4)||床',                  **surface['床'],                'area': 13.25},

    {'key': '子供室1->外部||S_外壁_一般部',        **surface['S_外壁_一般部'],     'area':  (7.1 - 3.2175) * 0.83},
    {'key': '子供室1->外部||S_外壁_熱橋部',        **surface['S_外壁_熱橋部'],     'area':  (7.1 - 3.2175) * 0.17},
    {'key': '子供室1->外部||S_窓_レース',          **surface['S_子供室1_窓'],      'area':  3.2175},
    {'key': '子供室1->寝室||間仕切壁',             **surface['間仕切壁'],          'area':  8.74},
    {'key': '子供室1->階間(L4)||床',               **surface['床'],                'area': 10.76},

    {'key': '子供室2->外部||E_外壁_一般部',        **surface['E_外壁_一般部'],     'area':  (8.74 - 0.66) * 0.83},
    {'key': '子供室2->外部||E_外壁_熱橋部',        **surface['E_外壁_熱橋部'],     'area':  (8.74 - 0.66) * 0.17},
    {'key': '子供室2->外部||E_窓_レース',          **surface['E_子供室2_窓'],      'area':  0.66},
    {'key': '子供室2->外部||S_外壁_一般部',        **surface['S_外壁_一般部'],     'area':  (7.1 - 3.2175) * 0.83},
    {'key': '子供室2->外部||S_外壁_熱橋部',        **surface['S_外壁_熱橋部'],     'area':  (7.1 - 3.2175) * 0.17},
    {'key': '子供室2->外部||S_窓_レース',          **surface['S_子供室2_窓'],      'area':  3.2175},
    {'key': '子供室2->子供室1||間仕切壁',          **surface['間仕切壁'],          'area':  8.74},
    {'key': '子供室2->階間(L4)||床',               **surface['床'],                'area': 10.77},

    {'key': '2階ホール->外部||E_外壁_一般部',      **surface['E_外壁_一般部'],     'area':  2.18 * 0.83},
    {'key': '2階ホール->外部||E_外壁_熱橋部',      **surface['E_外壁_熱橋部'],     'area':  2.18 * 0.17},
    {'key': '2階ホール->外部||N_外壁_一般部',      **surface['N_外壁_一般部'],     'area': (11.33 - 0.99) * 0.83},
    {'key': '2階ホール->外部||N_外壁_熱橋部',      **surface['N_外壁_熱橋部'],     'area': (11.33 - 0.99) * 0.17},
    {'key': '2階ホール->外部||N_窓',               **surface['N_2階ホール_窓'],    'area':  0.99},
    {'key': '2階ホール->ホール||天井',             **surface['天井'],              'area':  2.9},
    {'key': '2階ホール->クローゼット||間仕切壁',   **surface['間仕切壁'],          'area':  4.37},
    {'key': '2階ホール->寝室||間仕切壁',           **surface['間仕切壁'],          'area':  2.18 -1.422},
    {'key': '2階ホール->寝室||室内建具',           **surface['室内建具'],          'area':  1.422},
    {'key': '2階ホール->子供室1||間仕切壁',        **surface['間仕切壁'],          'area':  7.1 - 1.422},
    {'key': '2階ホール->子供室1||室内建具',        **surface['室内建具'],          'area':  1.422},
    {'key': '2階ホール->子供室2||間仕切壁',        **surface['間仕切壁'],          'area':  7.1 - 1.422},
    {'key': '2階ホール->子供室2||室内建具',        **surface['室内建具'],          'area':  1.422},
    {'key': '2階ホール->片流れ(3)||0_外壁_一般部', **surface['0_外壁_一般部'],     'area':  0.69 * 0.83},  #日射なし
    {'key': '2階ホール->片流れ(3)||0_外壁_熱橋部', **surface['0_外壁_熱橋部'],     'area':  0.69 * 0.17},  #日射なし
    {'key': '2階ホール->階間(L4)||床',             **surface['床'],                'area':  7.87},

    {'key': '切妻->外部||E_外壁_一般部',           **surface['E_外壁_一般部'],     'area':  (1.64 + 3.35) * 0.83},
    {'key': '切妻->外部||E_外壁_熱橋部',           **surface['E_外壁_熱橋部'],     'area':  (1.64 + 3.35) * 0.17},
    {'key': '切妻->外部||南面屋根',                **surface['南面屋根'],          'area': 28.6},
    {'key': '切妻->外部||北面屋根',                **surface['北面屋根'],          'area': 28.6},
    {'key': '切妻->外部||N_外壁_一般部',           **surface['N_外壁_一般部'],     'area':  2.87 * 0.83},
    {'key': '切妻->外部||N_外壁_熱橋部',           **surface['N_外壁_熱橋部'],     'area':  2.87 * 0.17},
    {'key': '切妻->外部||S_外壁_一般部',           **surface['S_外壁_一般部'],     'area':  2.87 * 0.83},
    {'key': '切妻->外部||S_外壁_熱橋部',           **surface['S_外壁_熱橋部'],     'area':  2.87 * 0.17},
    {'key': '切妻->外部||W_外壁_一般部',           **surface['W_外壁_一般部'],     'area':  (1.64 + 3.35) * 0.83},
    {'key': '切妻->外部||W_外壁_熱橋部',           **surface['W_外壁_熱橋部'],     'area':  (1.64 + 3.35) * 0.17},
    {'key': '切妻->クローゼット||外皮天井',        **surface['外皮天井'],          'area':  4.97},
    {'key': '切妻->寝室||外皮天井',                **surface['外皮天井'],          'area': 13.25},
    {'key': '切妻->子供室1||外皮天井',             **surface['外皮天井'],          'area': 10.76},
    {'key': '切妻->子供室2||外皮天井',             **surface['外皮天井'],          'area': 10.77},
    {'key': '切妻->2階ホール||外皮天井',           **surface['外皮天井'],          'area': 10.77},
    {'key': '切妻->2階トイレ||外皮天井',           **surface['外皮天井'],          'area':  1.66},

    {'key': '片流れ(3)->外部||E_外壁_一般部',      **surface['E_外壁_一般部'],     'area':  (0.58 + 0.27) * 0.83},
    {'key': '片流れ(3)->外部||E_外壁_熱橋部',      **surface['E_外壁_熱橋部'],     'area':  (0.58 + 0.27) * 0.17},
    {'key': '片流れ(3)->外部||屋根',               **surface['西面屋根'],          'area': 13.16},
    {'key': '片流れ(3)->外部||W_外壁_一般部',      **surface['W_外壁_一般部'],     'area':  (0.58 + 0.27) * 0.83},
    {'key': '片流れ(3)->外部||W_外壁_熱橋部',      **surface['W_外壁_熱橋部'],     'area':  (0.58 + 0.27) * 0.17},
    {'key': '片流れ(3)->外部||N_外壁_一般部',      **surface['N_外壁_一般部'],     'area':  1.02 * 0.83},
    {'key': '片流れ(3)->外部||N_外壁_熱橋部',      **surface['N_外壁_熱橋部'],     'area':  1.02 * 0.17},
    {'key': '片流れ(3)->台所||外皮天井',           **surface['外皮天井'],          'area':  4.14},
    {'key': '片流れ(3)->台所||0_外壁_一般部',      **surface['0_外壁_一般部'],     'area':  0.34 * 0.83},  #日射なし
    {'key': '片流れ(3)->台所||0_外壁_熱橋部',      **surface['0_外壁_熱橋部'],     'area':  0.34 * 0.17},  #日射なし
    {'key': '片流れ(3)->浴室||外皮天井',           **surface['外皮天井'],          'area':  3.31},
    {'key': '片流れ(3)->1階トイレ||外皮天井',      **surface['外皮天井'],          'area':  1.66},
    {'key': '片流れ(3)->洗面所||外皮天井',         **surface['外皮天井'],          'area':  3.31},
    {'key': '片流れ(3)->ホール||0_外壁_一般部',    **surface['0_外壁_一般部'],     'area':  (1.59 + 0.68) * 0.83},  #日射なし
    {'key': '片流れ(3)->ホール||0_外壁_熱橋部',    **surface['0_外壁_熱橋部'],     'area':  (1.59 + 0.68) * 0.17},  #日射なし

    {'key': '片流れ(4)->外部||0_外壁_一般部',      **surface['0_外壁_一般部'],     'area':  1.66 * 0.83},
    {'key': '片流れ(4)->外部||0_外壁_熱橋部',      **surface['0_外壁_熱橋部'],     'area':  1.66 * 0.17},
    {'key': '片流れ(4)->外部||屋根',               **surface['北面屋根2'],         'area':  5.66},
    {'key': '片流れ(4)->外部||S_外壁_一般部',      **surface['S_外壁_一般部'],     'area':  0.21 * 0.83},
    {'key': '片流れ(4)->外部||S_外壁_熱橋部',      **surface['S_外壁_熱橋部'],     'area':  0.21 * 0.17},
    {'key': '片流れ(4)->和室||外皮天井',           **surface['外皮天井'],          'area':  3.31},

    {'key': '階間(L4)->外部||E_外壁_一般部',       **surface['E_外壁_一般部'],     'area':  2.73 * 0.83},
    {'key': '階間(L4)->外部||E_外壁_熱橋部',       **surface['E_外壁_熱橋部'],     'area':  2.73 * 0.17},
    {'key': '階間(L4)->外部||N_外壁_一般部',       **surface['N_外壁_一般部'],     'area':  1.37 * 0.83},
    {'key': '階間(L4)->外部||N_外壁_熱橋部',       **surface['N_外壁_熱橋部'],     'area':  1.37 * 0.17},
    {'key': '階間(L4)->外部||S_外壁_一般部',       **surface['S_外壁_一般部'],     'area':  4.78 * 0.83},
    {'key': '階間(L4)->外部||S_外壁_熱橋部',       **surface['S_外壁_熱橋部'],     'area':  4.78 * 0.17},
    {'key': '階間(L4)->外部||W_外壁_一般部',       **surface['W_外壁_一般部'],     'area':  0.25 * 0.83},
    {'key': '階間(L4)->外部||W_外壁_熱橋部',       **surface['W_外壁_熱橋部'],     'area':  0.25 * 0.17},
    {'key': '階間(L4)->和室||天井',                **surface['天井'],              'area': 13.25},
    {'key': '階間(L4)->LD||天井',                  **surface['天井'],              'area': 21.53},
    {'key': '階間(L4)->台所||天井',                **surface['天井'],              'area':  4.14},
    {'key': '階間(L4)->ホール||天井',              **surface['天井'],              'area':  2.48 + 7.87},
    {'key': '階間(L4)->2階ホール||間仕切壁',       **surface['間仕切壁'],          'area':  0.46 + 1.59 + 0.46},
    {'key': '階間(L4)->片流れ(3)||0_外壁_一般部',  **surface['0_外壁_一般部'],     'area':  1.82 * 0.83},  #日射なし
    {'key': '階間(L4)->片流れ(3)||0_外壁_熱橋部',  **surface['0_外壁_熱橋部'],     'area':  1.82 * 0.17},  #日射なし
    {'key': '階間(L4)->片流れ(4)||0_外壁_一般部',  **surface['0_外壁_一般部'],     'area':  2.48 * 0.83},  #日射なし
    {'key': '階間(L4)->片流れ(4)||0_外壁_熱橋部',  **surface['0_外壁_熱橋部'],     'area':  2.48 * 0.17},  #日射なし

    {'key': '2階トイレ->外部||E_外壁_一般部',      **surface['E_外壁_一般部'],     'area':  2.18 * 0.83},
    {'key': '2階トイレ->外部||E_外壁_熱橋部',      **surface['E_外壁_熱橋部'],     'area':  2.18 * 0.17},
    {'key': '2階トイレ->外部||N_外壁_一般部',      **surface['N_外壁_一般部'],     'area':  (4.12 - 0.54) * 0.83},
    {'key': '2階トイレ->外部||N_外壁_熱橋部',      **surface['N_外壁_熱橋部'],     'area':  (4.12 - 0.54) * 0.17},
    {'key': '2階トイレ->外部||N_窓',               **surface['N_2階トイレ_窓'],    'area':  0.54},
    {'key': '2階トイレ->2階ホール||間仕切壁',      **surface['間仕切壁'],          'area':  4.37 + 2.18 -1.422},
    {'key': '2階トイレ->2階ホール||室内建具',      **surface['室内建具'],          'area':  1.422},
    {'key': '2階トイレ->片流れ(3)||0_外壁_一般部', **surface['0_外壁_一般部'],     'area':  0.25 * 0.83},  #日射なし
    {'key': '2階トイレ->片流れ(3)||0_外壁_熱橋部', **surface['0_外壁_熱橋部'],     'area':  0.25 * 0.17},  #日射なし
    {'key': '2階トイレ->階間(L4)||床',             **surface['床'],                'area':  1.66}
]

"""##発熱源"""

heat_ratio = {
    '人体':   {'convection': 0.50, 'radiation': 0.50},
    '蛍光灯': {'convection': 0.60, 'radiation': 0.40},
    '白熱灯': {'convection': 0.40, 'radiation': 0.60},
    '機器':   {'convection': 0.60, 'radiation': 0.40},
}

s_heat = vt.schedule.sensible_heat

input_data['heat_source'] = [
    {'key': 'LD_人体',        'room': 'LD',        'generation_rate': s_heat["LD"]["人体"],        **heat_ratio['人体']},
    {'key': 'LD_照明',        'room': 'LD',        'generation_rate': s_heat["LD"]["照明"],        **heat_ratio['蛍光灯']},
    {'key': 'LD_機器',        'room': 'LD',        'generation_rate': s_heat["LD"]["機器"],        **heat_ratio['機器']},
    {'key': '台所_照明',      'room': '台所',      'generation_rate': s_heat["台所"]["照明"],      **heat_ratio['蛍光灯']},
    {'key': '台所_機器1',     'room': '台所',      'generation_rate': s_heat["台所"]["機器1"],     **heat_ratio['機器']},
    {'key': '台所_機器2',     'room': '台所',      'generation_rate': s_heat["台所"]["機器2"],     **heat_ratio['機器']},
    {'key': '浴室_照明',      'room': '浴室',      'generation_rate': s_heat["浴室"]["照明"],      **heat_ratio['白熱灯']},
    {'key': '1階トイレ_照明', 'room': '1階トイレ', 'generation_rate': s_heat["1階トイレ"]["照明"], **heat_ratio['白熱灯']},
    {'key': '1階トイレ_機器', 'room': '1階トイレ', 'generation_rate': s_heat["1階トイレ"]["機器"], **heat_ratio['機器']},
    {'key': '洗面所_照明',    'room': '洗面所',    'generation_rate': s_heat["洗面所"]["照明"],    **heat_ratio['白熱灯']},
    {'key': '洗面所_機器',    'room': '洗面所',    'generation_rate': s_heat["洗面所"]["機器"],    **heat_ratio['機器']},
    {'key': 'ホール_照明1',   'room': 'ホール',    'generation_rate': s_heat["ホール"]["照明1"],   **heat_ratio['白熱灯']},
    {'key': 'ホール_照明2',   'room': 'ホール',    'generation_rate': s_heat["ホール"]["照明2"],   **heat_ratio['白熱灯']},
    {'key': '寝室_人体',      'room': '寝室',      'generation_rate': s_heat["寝室"]["人体"],      **heat_ratio['人体']},
    {'key': '寝室_照明',      'room': '寝室',      'generation_rate': s_heat["寝室"]["照明"],      **heat_ratio['蛍光灯']},
    {'key': '寝室_機器',      'room': '寝室',      'generation_rate': s_heat["寝室"]["機器"],      **heat_ratio['機器']},
    {'key': '子供室1_人体',   'room': '子供室1',   'generation_rate': s_heat["子供室1"]["人体"],   **heat_ratio['人体']},
    {'key': '子供室1_照明',   'room': '子供室1',   'generation_rate': s_heat["子供室1"]["照明"],   **heat_ratio['蛍光灯']},
    {'key': '子供室1_機器',   'room': '子供室1',   'generation_rate': s_heat["子供室1"]["機器"],   **heat_ratio['機器']},
    {'key': '子供室2_人体',   'room': '子供室2',   'generation_rate': s_heat["子供室2"]["人体"],   **heat_ratio['人体']},
    {'key': '子供室2_照明',   'room': '子供室2',   'generation_rate': s_heat["子供室2"]["照明"],   **heat_ratio['蛍光灯']},
    {'key': '子供室2_機器',   'room': '子供室2',   'generation_rate': s_heat["子供室2"]["機器"],   **heat_ratio['機器']}
]

h_heat = vt.schedule.latent_moisture

input_data['humidity_source'] = [
    {'key': 'LD_人体',        'room': 'LD',        'generation_rate': h_heat["LD"]["人体"]},
    {'key': '台所_機器',      'room': '台所',      'generation_rate': h_heat["台所"]["機器"]},
    {'key': '寝室_人体',      'room': '寝室',      'generation_rate': h_heat["寝室"]["人体"]},
    {'key': '子供室1_人体',   'room': '子供室1',   'generation_rate': h_heat["子供室1"]["人体"]},
    {'key': '子供室2_人体',   'room': '子供室2',   'generation_rate': h_heat["子供室2"]["人体"]},
]

"""##エアコン"""

highspec_ac = {
    "Q": {"cooling": {"min": 0.700, "rtd": 2.200, "max": 3.300},
          "heating": {"min": 0.700, "rtd": 2.500, "max": 5.400}},
    "P": {"cooling": {"min": 0.095, "rtd": 0.395, "max": 0.780},
          "heating": {"min": 0.095, "rtd": 0.390, "max": 1.360}},
    "V_inner": {"cooling": {"rtd": 12.1 / 60}, "heating": {"rtd": 13.1 / 60}},
    "V_outer": {"cooling": {"rtd": 28.2 / 60}, "heating": {"rtd": 25.5 / 60}}
}

standard_ac = {
    "Q": {"cooling": {"min": 0.900, "rtd": 2.200, "max": 2.800},
          "heating": {"min": 0.900, "rtd": 2.200, "max": 3.600}},
    "P": {"cooling": {"min": 0.170, "rtd": 0.455, "max": 0.745},
          "heating": {"min": 0.135, "rtd": 0.385, "max": 1.070}},
    "V_inner": {"cooling": {"rtd": 12.0 / 60}, "heating": {"rtd": 12.0 / 60}},
    "V_outer": {"cooling": {"rtd": 27.6 / 60}, "heating": {"rtd": 22.5 / 60}},
}

rac_ac ={
  "Q": {"cooling": { "rtd": 2.2, "max": 2.8 },
        "heating": { "rtd": 2.2, "max": 3.6 }},
  "P": {"cooling": { "rtd": 0.455 },
        "heating": { "rtd": 0.385 }},
  "dualcompressor": False
}

rac_standard_ac ={
  "Q": {"cooling": { "rtd": 5.60000, "max":  5.94462 },
        "heating": { "rtd": 6.68530, "max": 10.04705 }},
  "P": {"cooling": { "rtd": 5.60000 / 3.2432 },
        "heating": { "rtd": 6.68530 / 4.1573}},
  "dualcompressor": False
}

input_data["aircon"] = [
    {"key": "LDエアコン", "set": "LD", "in": "LD", "out": "LD", "outside": "外部",
                          "mode":     vt.schedule.ac_mode['region6']['LD'],
                          "pre_temp": vt.schedule.pre_tmp['region6']['LD'],
                          "model":    "RAC", "ac_spec":  rac_standard_ac
    },
    {"key": "寝室エアコン", "set": "寝室", "in": "寝室", "out": "寝室", "outside": "外部",
                            "mode":     vt.schedule.ac_mode['region6']['寝室'],
                            "pre_temp": vt.schedule.pre_tmp['region6']['寝室'],
                            "model":    "CRIEPI", "ac_spec":  highspec_ac
    },
    {"key": "子供室1エアコン", "set": "子供室1", "in": "子供室1", "out": "子供室1", "outside": "外部",
                               "mode":     vt.schedule.ac_mode['region6']['子供室1'],
                               "pre_temp": vt.schedule.pre_tmp['region6']['子供室1'],
                               "model":    "RAC", "ac_spec":  rac_ac
    },
    {"key": "子供室2エアコン", "set": "子供室2", "in": "子供室2", "out": "子供室2", "outside": "外部",
                               "mode":     vt.schedule.ac_mode['region6']['子供室2'],
                               "pre_temp": vt.schedule.pre_tmp['region6']['子供室2'],
                               "model":    "CRIEPI", "ac_spec":  highspec_ac
    },
]

"""#実行"""

from google.colab import userdata
base_url=userdata.get('VTSIMNX_API_URL')
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