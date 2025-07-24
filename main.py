from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from utils_f1 import enable_cache, load_artifacts, prepare_upcoming_features, predict_win_prob

CACHE_DIR = "fastf1_cache"
MODEL_PATH = "winner_model.pkl"
FEATS_PATH = "feats.parquet"
COLS_PATH  = "feats.cols.json"

enable_cache(CACHE_DIR)
model, feats_hist, feat_cols = load_artifacts(MODEL_PATH, FEATS_PATH, COLS_PATH)

app = FastAPI(title="F1 Predictor API")

class WhatIf(BaseModel):
    w_grid: float = 1.0
    w_form: float = 1.0
    w_team: float = 1.0
    manual_overrides: dict | None = None  # { "VER": {"grid_pos": 2, ...} }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(year: int = Query(..., ge=2014), round: int = Query(..., ge=1),
            w_grid: float = 1.0, w_form: float = 1.0, w_team: float = 1.0):
    try:
        base_df = prepare_upcoming_features(year, round, feats_hist)
    except Exception as e:
        raise HTTPException(400, str(e))
    df = base_df.copy()
    df["grid_pos"] *= w_grid
    for col in ["roll_avg_finish_3","roll_avg_grid_3","roll_wins_5","prev_points"]:
        df[col] *= w_form
    df["team_form_3"] *= w_team
    probs = predict_win_prob(model, feat_cols, df)
    df["win_prob"] = probs
    df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return df[["drv_abbr","team","grid_pos","win_prob"]].to_dict(orient="records")

@app.post("/predict/whatif")
def predict_whatif(year: int, round: int, payload: WhatIf):
    try:
        base_df = prepare_upcoming_features(year, round, feats_hist)
    except Exception as e:
        raise HTTPException(400, str(e))
    df = base_df.copy()
    df["grid_pos"] *= payload.w_grid
    for col in ["roll_avg_finish_3","roll_avg_grid_3","roll_wins_5","prev_points"]:
        df[col] *= payload.w_form
    df["team_form_3"] *= payload.w_team
    if payload.manual_overrides:
        for abbr, overrides in payload.manual_overrides.items():
            mask = df["drv_abbr"] == abbr
            for k,v in overrides.items():
                if k in df.columns:
                    df.loc[mask, k] = v
    probs = predict_win_prob(model, feat_cols, df)
    df["win_prob"] = probs
    df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)
    return df.to_dict(orient="records")
