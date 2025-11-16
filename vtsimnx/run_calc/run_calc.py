from typing import Dict, Any
import requests
import json

def run_calc(base_url: str, config_json: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/run"
    response = requests.post(url, json={"config": config_json})
    with open("calc_result.json", "w") as f:
        json.dump(response.json(), f, indent=4, ensure_ascii=False)
    return response.json()

if __name__ == "__main__":
    config_json = {
        "simulation": {
            "length": 8760,
            "timestep": 3600,
        }
    }
    calced_json = run(config_json)
    print(calced_json)