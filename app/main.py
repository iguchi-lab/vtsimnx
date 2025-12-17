"""
FastAPI ベースの VTSimNX API エンドポイント定義。

- /ping: ライブネス/ヘルスチェック
- /run: C++ ソルバを呼び出して結果 JSON を返す
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from app.solver_runner import run_solver

# API ルータ本体。OpenAPI のタイトルやバージョンをここで設定する。
app = FastAPI(title="VTSimNX API", version="0.1.0")

class SimulationRequest(BaseModel):
    """
    ソルバに渡す入力設定を表すデータモデル。

    - config: C++ ソルバが理解できる任意の設定ツリー（ネスト可）
    """
    config: Dict[str, Any]

class SimulationResponse(BaseModel):
    """
    ソルバの計算結果を表すデータモデル。

    - result: ソルバから返却される任意の JSON 互換オブジェクト
    """
    result: Dict[str, Any]

@app.get("/ping")
def ping():
    """軽量なヘルスチェック用エンドポイント。"""
    return {"status": "ok"}

@app.post("/run", response_model=SimulationResponse)
def run_simulation(req: SimulationRequest):
    """
    入力 JSON（config）を C++ ソルバに渡して、結果 JSON を返す。

    Args:
        req: クライアントから受け取った `SimulationRequest`。`config` を含む。

    Returns:
        `SimulationResponse`: ソルバの出力を `result` に格納して返す。

    Raises:
        HTTPException(500): ソルバ実行や結果読み取りで例外が発生した場合に返す。
    """
    try:
        output = run_solver(req.config)
    except Exception as e:
        # エラー時は 500 を返す
        raise HTTPException(status_code=500, detail=str(e))

    return SimulationResponse(result=output)
