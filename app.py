import os
import pandas as pd
import altair as alt
import streamlit as st

from utils_f1 import (
    enable_cache,
    load_artifacts,
    prepare_upcoming_features,
    predict_win_prob,
)

# ---------------- Page config ----------------
st.set_page_config(page_title="F1 Predictor", layout="wide")

# ---------------- Light CSS ------------------
st.markdown("""
<style>
:root{
  --bg:#f7f9fc;
  --card:#ffffff;
  --accent1:#008cff;
  --accent2:#00c56e;
  --text:#1c1e21;
  --sub:#6b6f76;
  --warn:#ffaf00;
  --radius:18px;
  --pad:1rem 1.2rem;
  --shadow:0 2px 8px rgba(0,0,0,0.06);
  --shadow2:0 0 18px rgba(0,140,255,0.18);
}
body, .stApp {background:var(--bg);}
h1,h2,h3,h4,h5,p,span,div {color:var(--text);}
section[data-testid="stSidebar"] {background: #eef1f6;}
.stTabs [data-baseweb="tab-list"] {gap: .75rem;}
.stTabs [data-baseweb="tab"] {
  background: var(--card);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
}
.stTabs [data-baseweb="tab"]:hover {background:#f0f2f5;}
[data-testid="stMarkdownContainer"] a {color: var(--accent1);}

.driver-card{
  background: var(--card);
  padding: var(--pad);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  text-align:center;
  position:relative;
}
.driver-card.primary{
  border:2px solid var(--warn);
  box-shadow:0 0 18px rgba(255,175,0,0.22);
}
.driver-card .abbr{font-size:1.8rem;font-weight:800;letter-spacing:.5px;}
.driver-card .team{color:var(--sub);margin-top:2px;font-size:0.9rem;}
.driver-card .conf{color:var(--accent2);font-size:1.0rem;margin-top:6px;}
.conf-bar{
  width:100%;height:6px;border-radius:3px;background:#e0e3e8;
  margin-top:6px;overflow:hidden;
}
.conf-fill{
  height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent1));
}
.big-btn{
  font-weight:600;padding:.9rem 1.2rem;border:none;border-radius:var(--radius);
  background:linear-gradient(90deg,var(--accent2),var(--accent1));
  color:#fff;width:100%;cursor:pointer;box-shadow:var(--shadow2);
}
.big-btn:active{transform:scale(0.99);}
.reset-btn{
  background:#dde2ea;color:var(--sub);border-radius:var(--radius);
  padding:.9rem 1.2rem;width:100%;border:1px solid #cbd2dc;
}
table td, table th {color:var(--text);}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ------------------
with st.sidebar:
    st.header("Artifacts")
    cache_dir = st.text_input("FastF1 cache dir", "fastf1_cache")
    model_path = st.text_input("Model (.pkl)", "winner_model.pkl")
    feats_path = st.text_input("Features (.parquet)", "feats.parquet")
    cols_path  = st.text_input("Columns (.json)", "feats.cols.json")

    st.header("Race")
    year = st.number_input("Year", 2014, 2100, 2025, 1)
    rnd  = st.number_input("Round", 1, 30, 12, 1)

    st.header("What‑if weights")
    defaults = {"w_grid":1.0,"w_form":1.0,"w_team":1.0}
    if "weights" not in st.session_state:
        st.session_state.weights = defaults.copy()

    w_grid = st.slider("Grid position", 0.1, 3.0, st.session_state.weights["w_grid"], 0.1)
    w_form = st.slider("Driver form", 0.1, 3.0, st.session_state.weights["w_form"], 0.1)
    w_team = st.slider("Team form", 0.1, 3.0, st.session_state.weights["w_team"], 0.1)

    c1, c2 = st.columns(2)
    with c1:
        update_click = st.button("Update Predictions", use_container_width=True, key="update")
    with c2:
        if st.button("Reset", use_container_width=True):
            st.session_state.weights = defaults.copy()
            st.rerun()

    manual_edit = st.checkbox("Manual feature editing")

# --------------- Load artifacts ---------------
@st.cache_resource(show_spinner=False)
def _load(mp, fp, cp):
    return load_artifacts(mp, fp, cp)

try:
    model, feats_hist, feat_cols = _load(model_path, feats_path, cols_path)
except Exception as e:
    st.exception(e)
    st.stop()

# --------------- Helpers ----------------------
def render_podium(df):
    podium = df.head(3).copy()
    order = [1,0,2] if len(podium)>=3 else range(len(podium))
    cols = st.columns(len(order))
    for idx, col in zip(order, cols):
        row = podium.iloc[idx]
        primary = "primary" if idx==1 else ""
        conf_pct = f"{row.win_prob:.1%}"
        width = int(row.win_prob*100)
        with col:
            st.markdown(f"""
            <div class="driver-card {primary}">
              <div class="abbr">{row.drv_abbr}</div>
              <div class="team">{row.team}</div>
              <div class="conf">{conf_pct} confidence</div>
              <div class="conf-bar"><div class="conf-fill" style="width:{width}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)

def chart(df):
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("drv_abbr:N", sort="-y", title="Driver"),
            y=alt.Y("win_prob:Q", title="P(win)", axis=alt.Axis(format='%')),
            tooltip=["drv_abbr","team","grid_pos",alt.Tooltip("win_prob:Q", format=".2%")]
        ).properties(height=360)
    )

# --------------- Main -------------------------
st.title("F1 Winner Predictor (Light UI)")

if update_click:
    st.session_state.weights = {"w_grid":w_grid,"w_form":w_form,"w_team":w_team}
    enable_cache(cache_dir)

    try:
        base_df = prepare_upcoming_features(int(year), int(rnd), feats_hist)
    except Exception as e:
        st.exception(e)
        st.stop()

    df = base_df.copy()
    df["grid_pos"]              *= w_grid
    df["roll_avg_finish_3"]     *= w_form
    df["roll_avg_grid_3"]       *= w_form
    df["roll_wins_5"]           *= w_form
    df["prev_points"]           *= w_form
    df["team_form_3"]           *= w_team

    if manual_edit:
        st.info("Edit values then hit Update again")
        edit_cols = ["grid_pos","roll_avg_finish_3","roll_avg_grid_3","roll_wins_5","prev_points","team_form_3"]
        editable = df[["drv_abbr","team"]+edit_cols].copy()
        edited = st.data_editor(editable, num_rows="fixed", use_container_width=True, key="editor")
        for c in edit_cols:
            df[c] = edited[c].values

    probs = predict_win_prob(model, feat_cols, df)
    df["win_prob"] = probs
    df = df.sort_values("win_prob", ascending=False).reset_index(drop=True)

    st.header("🏆 Podium Predictions")
    render_podium(df)

    st.subheader("Full Probability Distribution")
    st.altair_chart(chart(df), use_container_width=True)

    st.subheader("Prediction Table")
    st.dataframe(df[["drv_abbr","team","grid_pos","win_prob"]], use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", data=csv, file_name=f"prediction_{year}_R{rnd:02d}.csv", mime="text/csv")

    with st.expander("Debug / Paths"):
        st.write("cwd:", os.getcwd())
        st.json({"model_path":model_path, "feats_path":feats_path, "cols_path":cols_path, "cache_dir":cache_dir})
else:
    st.write("Set parameters on the left, then click **Update Predictions**.")
