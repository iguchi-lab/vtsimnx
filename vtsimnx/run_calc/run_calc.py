from typing import Dict, Any
import requests
import os

def run_calc(config_json: Dict[str, Any]) -> Dict[str, Any]:
    base_url = os.getenv("VTSIMNX_API_URL")
    if not base_url:
        raise ValueError("VTSIMNX_API_URL is not set")
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