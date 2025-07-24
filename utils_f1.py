import json
import joblib
import pandas as pd
import fastf1
from pathlib import Path


def enable_cache(cache_dir: str = "fastf1_cache"):
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(p.as_posix())


def load_artifacts(model_path="winner_model.pkl",
                   feats_path="feats.parquet",
                   cols_path="feats.cols.json"):
    mp, fp, cp = map(Path, [model_path, feats_path, cols_path])

    if not mp.exists():
        raise FileNotFoundError(f"Model not found: {mp}")
    if not fp.exists():
        raise FileNotFoundError(f"Features parquet not found: {fp}")
    if not cp.exists():
        raise FileNotFoundError(f"Columns json not found: {cp}")

    model = joblib.load(mp)
    try:
        feats = pd.read_parquet(fp)  # requires pyarrow or fastparquet
    except Exception as e:
        raise RuntimeError(f"Failed to read {fp}: {e}")
    feat_cols = json.loads(cp.read_text())

    return model, feats, feat_cols


def prepare_upcoming_features(year: int, rnd: int, df_hist: pd.DataFrame) -> pd.DataFrame:
    ses = fastf1.get_session(year, rnd, "R")
    ses.load(laps=False, telemetry=False, weather=False)
    res = ses.results.copy()

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    ev_row = schedule.set_index("RoundNumber").loc[rnd]

    res["Year"] = year
    res["Round"] = rnd
    res["EventName"] = ev_row["EventName"]
    res["Date"] = pd.to_datetime(ev_row["EventDate"])

    res.rename(
        columns={
            "DriverId": "driver_id",
            "Abbreviation": "drv_abbr",
            "BroadcastName": "drv_name",
            "TeamName": "team",
            "GridPosition": "grid_pos",
            "Points": "points",
        },
        inplace=True,
    )

    needed_cols = [
        "driver_id",
        "roll_avg_finish_3",
        "roll_avg_grid_3",
        "roll_wins_5",
        "prev_points",
    ]
    driver_latest = (
        df_hist.sort_values("Date")
        .groupby("driver_id")
        .tail(1)[needed_cols]
        .set_index("driver_id")
    )
    res = res.join(driver_latest, on="driver_id")

    team_latest = (
        df_hist.sort_values("Date")
        .groupby("team")
        .tail(1)[["team", "team_form_3"]]
        .set_index("team")
    )
    res = res.join(team_latest, on="team")

    res["is_sprint"] = res["EventName"].str.contains("Sprint", case=False, na=False).astype(int)

    # rookies / new teams defaults
    res["roll_avg_finish_3"] = res["roll_avg_finish_3"].fillna(df_hist["finish_pos"].mean())
    res["roll_avg_grid_3"] = res["roll_avg_grid_3"].fillna(df_hist["grid_pos"].mean())
    res["roll_wins_5"] = res["roll_wins_5"].fillna(0)
    res["prev_points"] = res["prev_points"].fillna(0)
    res["team_form_3"] = res["team_form_3"].fillna(df_hist["points"].mean())

    return res


def predict_win_prob(model, feat_cols, df_features):
    proba = model.predict_proba(df_features[feat_cols])[:, 1]
    return proba
