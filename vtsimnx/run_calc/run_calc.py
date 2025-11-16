from typing import Dict, Any
import requests
import os

def run_calc(base_url: str, config_json: Dict[str, Any]) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/run"
    response = requests.post(url, json={"config": config_json})
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