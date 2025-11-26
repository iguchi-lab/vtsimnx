from typing import Dict, Any, Optional
import requests
import json

def run_calc(base_url: str, config_json: Dict[str, Any], output_path: Optional[str] = "calc_result.json") -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/run"
    response = requests.post(url, json={"config": config_json})
    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(response.json(), f, indent=4, ensure_ascii=False)
    return response.json()

if __name__ == "__main__":
    config_json = {
        "simulation": {
            "length": 8760,
            "timestep": 3600,
        }
    }
    base_url = "http://localhost:8000"
    calced_json = run_calc(base_url, config_json)
    print(calced_json)