#!/usr/bin/env python3
"""
mymodel.py  —  Train and use an F1 winner predictor with FastF1.

Examples
--------
1) Train and evaluate (saves artifacts):
   python mymodel.py --train-start 2018 --train-end 2024 --test-year-start 2023

2) Predict a race using the saved model (no retrain):
   python mymodel.py --predict-year 2025 --predict-round 12 --reuse-model

3) Force retrain even if artifacts exist:
   python mymodel.py --train-start 2018 --train-end 2024 --force-train

Artifacts written
-----------------
winner_model.pkl        # sklearn Pipeline
feats.parquet           # engineered historical feature table
feats.cols.json         # ordered list of feature column names
"""

import argparse
import json
import pathlib
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier

import fastf1


# ----------------------- Cache -----------------------
def enable_cache(cache_dir: str = "fastf1_cache"):
    p = pathlib.Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(p.as_posix())


# ----------------------- Data ------------------------
def collect_results(year_start: int, year_end: int, load_telemetry: bool = False) -> pd.DataFrame:
    frames = []
    for year in range(year_start, year_end + 1):
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        for _, ev in schedule.iterrows():
            rnd = int(ev["RoundNumber"])
            ses = fastf1.get_session(year, rnd, "R")
            ses.load(laps=load_telemetry, telemetry=load_telemetry, weather=False)
            res = ses.results.copy()

            res["Year"] = year
            res["Round"] = rnd
            res["EventName"] = ev["EventName"]
            res["Date"] = pd.to_datetime(ev["EventDate"])

            res.rename(
                columns={
                    "DriverId": "driver_id",
                    "Abbreviation": "drv_abbr",
                    "BroadcastName": "drv_name",
                    "TeamName": "team",
                    "Position": "finish_pos",
                    "GridPosition": "grid_pos",
                    "Points": "points",
                },
                inplace=True,
            )
            frames.append(res)

    df = pd.concat(frames, ignore_index=True)
    df = df[df.finish_pos.notna()]
    df["finish_pos"] = df["finish_pos"].astype(int)
    df["grid_pos"] = df["grid_pos"].astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "Date"]).reset_index(drop=True)

    def per_driver(g):
        g = g.sort_values("Date")
        g["roll_avg_finish_3"] = g["finish_pos"].rolling(3, min_periods=1).mean().shift(1)
        g["roll_avg_grid_3"] = g["grid_pos"].rolling(3, min_periods=1).mean().shift(1)
        g["roll_wins_5"] = (g["finish_pos"] == 1).rolling(5, min_periods=1).sum().shift(1)
        g["prev_points"] = g["points"].shift(1)
        return g

    df = df.groupby("driver_id", group_keys=False).apply(per_driver)

    def per_team(g):
        g = g.sort_values("Date")
        g["team_form_3"] = g["points"].rolling(3, min_periods=1).mean().shift(1)
        return g

    df = df.groupby("team", group_keys=False).apply(per_team)

    df["is_sprint"] = df["EventName"].str.contains("Sprint", case=False, na=False).astype(int)
    df["winner"] = (df["finish_pos"] == 1).astype(int)

    df = df.dropna(
        subset=["roll_avg_finish_3", "roll_avg_grid_3", "team_form_3", "prev_points"]
    ).reset_index(drop=True)
    return df


# ----------------------- Model -----------------------
def build_model():
    num_cols = [
        "grid_pos",
        "roll_avg_finish_3",
        "roll_avg_grid_3",
        "roll_wins_5",
        "prev_points",
        "team_form_3",
    ]
    cat_cols = ["driver_id", "team", "EventName"]

    pre = ColumnTransformer(
        [
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("prep", pre), ("clf", clf)])
    return pipe, num_cols + cat_cols


def chronological_split(df: pd.DataFrame, test_year_start: int | None):
    if test_year_start is None:
        cutoff = df["Date"].quantile(0.8)
        train_df = df[df["Date"] < cutoff]
        test_df = df[df["Date"] >= cutoff]
    else:
        train_df = df[df["Year"] < test_year_start]
        test_df = df[df["Year"] >= test_year_start]
    return train_df, test_df


def train_and_eval(feats: pd.DataFrame, test_year_start: int | None):
    train_df, test_df = chronological_split(feats, test_year_start)
    pipe, feat_cols = build_model()

    X_train = train_df[feat_cols]
    y_train = train_df["winner"]
    X_test = test_df[feat_cols]
    y_test = test_df["winner"]

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    try:
        auc = roc_auc_score(y_test, proba)
    except ValueError:
        auc = np.nan

    print(f"Test Accuracy: {acc:.3f}")
    print(f"Test ROC-AUC:  {auc:.3f}")
    return pipe, feat_cols


# ------------------- Prediction ----------------------
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

    # defaults for rookies / new teams
    res["roll_avg_finish_3"] = res["roll_avg_finish_3"].fillna(df_hist["finish_pos"].mean())
    res["roll_avg_grid_3"] = res["roll_avg_grid_3"].fillna(df_hist["grid_pos"].mean())
    res["roll_wins_5"] = res["roll_wins_5"].fillna(0)
    res["prev_points"] = res["prev_points"].fillna(0)
    res["team_form_3"] = res["team_form_3"].fillna(df_hist["points"].mean())

    return res


# ----------------------- Main ------------------------
def save_artifacts(model, feats, feat_cols, model_path, feats_path, cols_path):
    joblib.dump(model, model_path)
    feats.to_parquet(feats_path)
    pathlib.Path(cols_path).write_text(json.dumps(feat_cols))
    print(f"Saved model -> {pathlib.Path(model_path).resolve()}")
    print(f"Saved feats -> {pathlib.Path(feats_path).resolve()}")
    print(f"Saved cols  -> {pathlib.Path(cols_path).resolve()}")


def load_artifacts(model_path, feats_path, cols_path):
    mp, fp, cp = map(pathlib.Path, [model_path, feats_path, cols_path])
    if not mp.exists() or not fp.exists() or not cp.exists():
        raise FileNotFoundError("Artifacts missing")
    model = joblib.load(mp)
    feats = pd.read_parquet(fp)
    feat_cols = json.loads(cp.read_text())
    return model, feats, feat_cols


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/predict F1 winner model")
    parser.add_argument("--cache-dir", type=str, default="fastf1_cache")

    # training inputs
    parser.add_argument("--train-start", type=int)
    parser.add_argument("--train-end", type=int)
    parser.add_argument("--test-year-start", type=int, default=None)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--force-train", action="store_true", help="Ignore existing artifacts")

    # prediction inputs
    parser.add_argument("--predict-year", type=int)
    parser.add_argument("--predict-round", type=int)
    parser.add_argument("--out-csv", type=str, default=None)

    # artifact paths
    parser.add_argument("--model-path", type=str, default="winner_model.pkl")
    parser.add_argument("--feats-path", type=str, default="feats.parquet")
    parser.add_argument("--cols-path", type=str, default="feats.cols.json")

    # reuse flag
    parser.add_argument("--reuse-model", action="store_true",
                        help="Load existing artifacts if present; skip training")

    args = parser.parse_args()
    enable_cache(args.cache_dir)

    mp, fp, cp = args.model_path, args.feats_path, args.cols_path

    need_training = args.force_train
    if not need_training:
        if args.reuse_model:
            # only load if all exist
            need_training = not (pathlib.Path(mp).exists() and pathlib.Path(fp).exists() and pathlib.Path(cp).exists())
        else:
            # user provided train years => we train
            need_training = args.train_start is not None and args.train_end is not None

    if need_training:
        if args.train_start is None or args.train_end is None:
            raise SystemExit("Training requested but --train-start/--train-end not provided")
        print("Collecting results")
        raw = collect_results(args.train_start, args.train_end, load_telemetry=args.telemetry)

        print("Engineering features")
        feats = engineer_features(raw)

        print("Training")
        model, feat_cols = train_and_eval(feats, args.test_year_start)

        print("Persisting artifacts")
        save_artifacts(model, feats, feat_cols, mp, fp, cp)
    else:
        print("Loading existing artifacts")
        model, feats, feat_cols = load_artifacts(mp, fp, cp)

    # Prediction block (optional)
    if args.predict_year and args.predict_round:
        print(f"Predicting {args.predict_year} Round {args.predict_round}")
        upcoming = prepare_upcoming_features(args.predict_year, args.predict_round, feats)
        probs = model.predict_proba(upcoming[feat_cols])[:, 1]
        upcoming["win_prob"] = probs
        out = upcoming.sort_values("win_prob", ascending=False)

        print(out[["drv_abbr", "team", "grid_pos", "win_prob"]].to_string(index=False))

        if args.out_csv:
            out.to_csv(args.out_csv, index=False)
            print(f"Saved prediction table -> {pathlib.Path(args.out_csv).resolve()}")
