import numpy as np

from vtsimnx.builder.surfaces import process_surfaces


def test_process_surfaces_adds_nocturnal_branch():
    surface = {
        "key": "室内->外部",
        "part": "wall",
        "area": 2.0,
        "u_value": 1.0,
        # 1ステップあたりの夜間放射量（例）[W/m2]
        "nocturnal": [10.0, 20.0, 30.0],
    }

    nodes, branches = process_surfaces([surface], sim_length=3, add_solar=False, add_radiation=False)
    noct = [b for b in branches if b.get("subtype") == "nocturnal_loss"]
    assert len(noct) == 1
    b = noct[0]

    # void->外部側表面ノード へ、負の heat_generation（表面からの流出）として入る
    assert b["key"].startswith("void->外部-室内_wall_s")
    hg = np.array(b["heat_generation"], dtype=float)
    assert hg.shape == (3,)
    assert hg.tolist() == [-20.0, -40.0, -60.0]  # -area * nocturnal


