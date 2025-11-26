import json
import vtsimnx as vt

wind_speed   = 0.0        #[m/s]
rho_air      = 1.205      #[kg/m3]
a0           = 0.2748 / 3600    #単位長さ通気率 [m3/(h･Pa^1/n)/m]
n0           = 1.81

C_East_1F    = -0.4
C_East_2F    = -0.5
C_South_1F   = +0.6
C_South_2F   = +0.8
C_West_1F    = -0.4
C_West_2F    = -0.5
C_North_1F   = -0.2
C_North_2F   = -0.3

def devided_opening(node1, node2, door, w, h, height, area, alpha=0.65, divs=10):
    opening_vertical = [{
        "key": node1 + " -> " + node2 + " || " + door + " W=" + str(w) + ", H=" + str(h) + " 垂直　下から" + str(n),
        "type": "simple_opening",
        "alpha": alpha,
        "area": area / divs,
        "h_from": height + h / divs * (n - 1 / 2),
        "h_to": height + h / divs * (n - 1 / 2)}
        for n in range(1, divs + 1)
    ]
    return opening_vertical

def devided_gap(node1, node2, window, w, h, height, a0=a0, n0=n0, divs=10, number=3):
    gap_lower = {
        "key": node1 + " -> " + node2 + " || " + window + " W=" + str(w) + ", H=" + str(h) + " 下端",
        "type": "gap",
        "a": a0 * w,
        "n": n0,
        "h_from": height,
        "h_to": height
    }
    gap_vertical = [{
        "key": node1 + " -> " + node2 + " || " + window + " W=" + str(w) + ", H=" + str(h) + " 垂直 下から" + str(n),
        "type": "gap",
        "a": a0 * h / divs * number,
        "n": n0,
        "h_from": height + h / divs * (n - 1 / 2),
        "h_to": height + h / divs * (n - 1 / 2)}
        for n in range(1, divs + 1)
    ]
    gap_upper = {
        "key": node1 + " -> " + node2 + " || " + window + " W=" + str(w) + ", H=" + str(h) + " 上端",
        "type": "gap",
        "a": a0 * w,
        "n": n0,
        "h_from": height + h,
        "h_to": height + h
    }
    output_list = []
    output_list.append(gap_lower)
    output_list.extend(gap_vertical)
    output_list.append(gap_upper)
    return output_list

#window_gap("南1F", "室内", "和室", 2.550, 1.80, 0.510)

def gap_all(node1, node2, window, w, h, height, a0=a0, n0=n0, divs=10, number=3):
    gap = {
        "key": node1 + " -> " + node2 + " || " + window + " W=" + str(w) + ", H=" + str(h),
        "type": "gap",
        "a": a0 * (w * 2+ h * number),
        "n": n0,
        "h_from": height + h / 2,
        "h_to": height + h / 2
    }
    output_list = [gap]
    return output_list

#window_gap_all("南1F", "室内", "和室", 2.550, 1.80, 0.510)

def main():

    input_data = {
        "simulation": {
            "index":     {"start": "2025-08-24 00:00:00", "end":   "2025-08-24 01:00:00",
                            "timestep": 60, "length": 1},
            "tolerance": {"ventilation": 1e-6, "thermal": 1e-6, "convergence": 1e-6}
        },
        "nodes": [
            {"key": "東1F",         "p": 1/2 * rho_air * wind_speed**2 * C_East_1F,  "t":  0.0},
            {"key": "東2F",         "p": 1/2 * rho_air * wind_speed**2 * C_East_2F,  "t":  0.0},
            {"key": "南1F",         "p": 1/2 * rho_air * wind_speed**2 * C_South_1F, "t":  0.0},
            {"key": "南2F",         "p": 1/2 * rho_air * wind_speed**2 * C_South_2F, "t":  0.0},
            {"key": "西1F",         "p": 1/2 * rho_air * wind_speed**2 * C_West_1F,  "t":  0.0},
            {"key": "西2F",         "p": 1/2 * rho_air * wind_speed**2 * C_West_2F,  "t":  0.0},
            {"key": "北1F",         "p": 1/2 * rho_air * wind_speed**2 * C_North_1F, "t":  0.0},
            {"key": "北2F",         "p": 1/2 * rho_air * wind_speed**2 * C_North_2F, "t":  0.0},
            {"key": "和室",         "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "LDK",          "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "浴室",         "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "1Fトイレ",     "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "洗面所",       "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "1・2Fホール",  "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "クローゼット", "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "寝室",         "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "子供室1",      "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "子供室2",      "p": 0.0,                                        "t": 20.0, "calc_p": True},
            {"key": "2Fトイレ",     "p": 0.0,                                        "t": 20.0, "calc_p": True},
        ],
        "ventilation_branches":
            devided_gap("南1F", "和室",         "和室",         2.550, 1.800, 0.510) +
            devided_gap("南1F", "LDK",          "LDK",          1.650, 2.100, 0.510) +
            devided_gap("南1F", "LDK",          "LDK",          1.650, 2.100, 0.510) +
            devided_gap("東1F", "LDK",          "LDK",          1.350, 1.300, 1.010) +
            devided_gap("東1F", "LDK",          "LDK",          1.400, 0.700, 1.610) +
            devided_gap("北1F", "LDK",          "LDK",          0.900, 1.800, 0.510) +
            devided_gap("北1F", "浴室",         "浴室",         0.600, 0.900, 1.410) +
            devided_gap("北1F", "1Fトイレ",     "1Fトイレ",     0.600, 0.900, 1.410) +
            devided_gap("北1F", "洗面所",       "洗面所",       0.600, 0.900, 1.410) +
            devided_gap("北1F", "1・2Fホール",  "1・2Fホール",  0.600, 0.900, 1.410) +
            devided_gap("西1F", "1・2Fホール",  "1・2Fホール",  0.900, 2.100, 0.000) +
            devided_gap("西2F", "クローゼット", "クローゼット", 0.600, 0.900, 4.310) +
            devided_gap("南2F", "寝室",         "寝室",         1.650, 1.050, 4.310) +
            devided_gap("西2F", "寝室",         "寝室",         0.900, 1.100, 4.310) +
            devided_gap("南2F", "子供室1",      "子供室1",      1.650, 1.950, 3.410) +
            devided_gap("南2F", "子供室2",      "子供室2",      1.650, 1.950, 3.410) +
            devided_gap("東2F", "子供室2",      "子供室2",      0.600, 1.100, 4.310) +
            devided_gap("北2F", "1・2Fホール",  "1・2Fホール",  0.900, 1.100, 4.447) +
            devided_gap("北2F", "2Fトイレ",     "2Fトイレ",     0.600, 0.900, 4.447) +

            devided_gap("LDK",          "和室",        "LDK-和室",             1.800, 1.800, 0.510, a0=84.94/3600/(1.800*2+1.800*3), n0=1.56) +
            devided_gap("和室",         "1・2Fホール", "和室-1・2Fホール",     0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("LDK",          "1・2Fホール", "LDK-1・2Fホール_1",    0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("LDK",          "1・2Fホール", "LDK-1・2Fホール_2",    0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("LDK",          "1・2Fホール", "LDK-1・2Fホール_3",    0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("浴室",         "洗面所",      "浴室-洗面所",          0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "1Fトイレ",    "LDK-1Fトイレ",         0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "洗面所",      "LDK-洗面所",           0.790, 1.800, 0.510, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("クローゼット", "寝室",        "クローゼット-寝室",    0.790, 1.800, 3.410, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "寝室",        "1・2Fホール-寝室",     0.790, 1.800, 3.410, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "子供室1",     "1・2Fホール-子供室1",  0.790, 1.800, 3.410, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "子供室2",     "1・2Fホール-子供室2",  0.790, 1.800, 3.410, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2) +
            devided_gap("1・2Fホール",  "2Fトイレ",    "1・2Fホール-2Fトイレ", 0.790, 1.800, 3.410, a0=65.48/3600/(0.790*2+1.800*2), n0=1.74, number=2)
    }

    config_json = vt.build_config(input_data)

    with open('neutoral_close.json', 'w') as f:
        json.dump(config_json, f, indent=4)


if __name__ == "__main__":
    main()