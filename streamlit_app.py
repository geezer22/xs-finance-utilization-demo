# streamlit_app.py
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Page setup ----------
st.set_page_config(page_title="XS Finance & Utilization Demo", layout="wide")
st.title("XSOLIS-style Finance & Utilization Demo (Synthetic)")

# ---------- Data loader (robust for cloud) ----------
@st.cache_data
def load_data():
    """
    - Resolve path relative to this file
    - Prefer cases_enriched.csv; else fall back to cases.csv and enrich on the fly
    - Show a tiny debug caption so we can see what the container sees
    """
    base = Path(__file__).resolve().parent
    data_dir = base / "data"

    # If some hosts mount differently, try CWD as fallback
    alt_dir = Path.cwd() / "data"
    if not data_dir.exists() and alt_dir.exists():
        data_dir = alt_dir

    # Debug: which CSVs are present?
    try:
        csvs_present = sorted(p.name for p in data_dir.glob("*.csv"))
    except Exception:
        csvs_present = []
    st.caption(f"Data dir: {data_dir} | exists: {data_dir.exists()} | files: {csvs_present}")

    enriched_path = data_dir / "cases_enriched.csv"
    raw_path      = data_dir / "cases.csv"

    if enriched_path.exists():
        df = pd.read_csv(enriched_path, parse_dates=["admit_date", "discharge_date"])
        return df, True

    if raw_path.exists():
        df = pd.read_csv(raw_path, parse_dates=["admit_date", "discharge_date"])
        df = enrich(df)   # create needed columns so the app can run
        return df, False

    raise FileNotFoundError(
        f"Could not find {enriched_path} or {raw_path}. "
        f"Ensure the repo includes a /data folder with one of those files."
    )

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal enrichment so the app works even if only cases.csv exists.

    Creates:
      - prov_budget_pct / prov_payer_pct (+ clipped versions)
      - robust-scaled LOS (los_norm)
      - simple risk_score (0–100) + priority_bucket
      - mismatch flag if not present (abs(provider-payer)/payer > 5%)
      - admit_month if not present
    """
    df = df.copy()

    # Provider vs Budget / Payer deltas
    if {"provider_cost", "budget_cost"} <= set(df.columns):
        df["prov_budget_pct"] = np.where(
            df["budget_cost"] > 0,
            (df["provider_cost"] - df["budget_cost"]) / df["budget_cost"],
            0.0,
        )
    else:
        df["prov_budget_pct"] = 0.0

    if {"provider_cost", "payer_cost"} <= set(df.columns):
        df["prov_payer_pct"] = np.where(
            df["payer_cost"] > 0,
            (df["provider_cost"] - df["payer_cost"]) / df["payer_cost"],
            0.0,
        )
    else:
        df["prov_payer_pct"] = 0.0

    # Mismatch flag if not provided
    if "mismatch" not in df.columns:
        df["mismatch"] = (df["prov_payer_pct"].abs() > 0.05).astype(int)

    # Robust LOS scaling
    if "length_of_stay" in df.columns:
        los = pd.to_numeric(df["length_of_stay"], errors="coerce")
        med = los.median()
        mad = (los - med).abs().median()
        mad = mad if mad and np.isfinite(mad) else 1.0
        df["los_norm"] = (los - med) / (1.4826 * mad)
    else:
        df["los_norm"] = 0.0

    # Clips & defaults
    df["prov_budget_pct_clip"] = df["prov_budget_pct"].clip(-0.5, 2.0)
    df["prov_payer_pct_clip"]  = df["prov_payer_pct"].clip(-0.5, 2.0)

    if "readmission_30d" not in df.columns:
        df["readmission_30d"] = 0
    if "drg_severity" not in df.columns:
        df["drg_severity"] = 0.5  # neutral

    # Simple monotonic risk → sigmoid to 0–100
    linear = (
        0.30 * df["los_norm"].clip(-2, 5)
        + 0.25 * df["prov_budget_pct_clip"]
        + 0.20 * df["prov_payer_pct_clip"]
        + 0.15 * df["readmission_30d"]
        + 0.10 * df["drg_severity"]
    )
    df["risk_score"] = (1.0 / (1.0 + np.exp(-3.0 * (linear - 0.0))) * 100).round(1)
    df["priority_bucket"] = pd.cut(
        df["risk_score"], bins=[-0.1, 40, 70, 100], labels=["Low", "Medium", "High"]
    )

    # Admit month if not present
    if "admit_month" not in df.columns and "admit_date" in df.columns:
        df["admit_month"] = df["admit_date"].dt.to_period("M").dt.to_timestamp()

    return df

# Load data
df, was_enriched_file = load_data()

# ---------- Sidebar filters ----------
def safe_unique(col):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

providers = ["All"] + safe_unique("provider")
payers    = ["All"] + safe_unique("payer")
drgs      = ["All"] + safe_unique("drg")

p_sel  = st.sidebar.selectbox("Provider", providers, index=0)
pay_sel= st.sidebar.selectbox("Payer",    payers,    index=0)
drg_sel= st.sidebar.selectbox("DRG",      drgs,      index=0)

mask = np.ones(len(df), dtype=bool)
if p_sel  != "All" and "provider" in df.columns: mask &= df["provider"].eq(p_sel)
if pay_sel!= "All" and "payer"    in df.columns: mask &= df["payer"].eq(pay_sel)
if drg_sel!= "All" and "drg"      in df.columns: mask &= df["drg"].eq(drg_sel)
dff = df.loc[mask].copy()

# ---------- KPIs ----------
c1, c2, c3, c4, c5 = st.columns(5)

def pct(x):
    try:    return f"{x:.1%}"
    except: return "—"

def fnum(x):
    try:    return f"{x:,.0f}"
    except: return "—"

c1.metric("Cases", fnum(len(dff)))

los_mean = dff["length_of_stay"].mean() if "length_of_stay" in dff.columns else np.nan
c2.metric("Avg LOS (days)", f"{los_mean:.2f}" if np.isfinite(los_mean) else "—")

readmit = dff["readmission_30d"].mean() if "readmission_30d" in dff.columns else np.nan
c3.metric("Readmit Rate", pct(readmit) if readmit == readmit else "—")

mismatch = (dff["mismatch"].mean() if "mismatch" in dff.columns else None)
if mismatch is None and "prov_payer_pct" in dff.columns:
    mismatch = (dff["prov_payer_pct"].abs() > 0.05).mean()
c4.metric("Mismatch Rate", pct(mismatch) if mismatch == mismatch else "—")

risk_mean = dff["risk_score"].mean() if "risk_score" in dff.columns else np.nan
c5.metric("Avg Risk Score", f"{risk_mean:.1f}" if np.isfinite(risk_mean) else "—")

# ---------- Chart 1: LOS trend ----------
st.subheader("Average LOS by Month")
if "admit_month" not in dff.columns and "admit_date" in dff.columns:
    dff["admit_month"] = dff["admit_date"].dt.to_period("M").dt.to_timestamp()

if {"admit_month", "length_of_stay"} <= set(dff.columns):
    los_by_month = dff.groupby("admit_month")["length_of_stay"].mean()
else:
    los_by_month = pd.Series(dtype=float)

fig1, ax1 = plt.subplots()
los_by_month.plot(ax=ax1)
ax1.set_title("Average Length of Stay by Month")
ax1.set_xlabel("Month")
ax1.set_ylabel("Avg LOS (days)")
ax1.grid(True, alpha=0.2)
fig1.tight_layout()
st.pyplot(fig1)

# ---------- Chart 2: Mismatch rate by provider ----------
st.subheader("Mismatch Rate by Provider")
if "mismatch" in dff.columns and "provider" in dff.columns:
    mm = dff.groupby("provider")["mismatch"].mean().sort_values()
elif {"provider", "prov_payer_pct"} <= set(dff.columns):
    mm = dff.groupby("provider")["prov_payer_pct"].apply(lambda s: (s.abs() > 0.05).mean()).sort_values()
else:
    mm = pd.Series(dtype=float)

fig2, ax2 = plt.subplots(figsize=(8, 4))
mm.plot(kind="barh", ax=ax2)
ax2.set_xlabel("Mismatch Rate")
ax2.set_ylabel("Provider")
ax2.set_title("Mismatch Rate by Provider")
fig2.tight_layout()
st.pyplot(fig2)

# ---------- Chart 3: Risk distribution ----------
st.subheader("Risk Score Distribution")
if "risk_score" in dff.columns:
    fig3, ax3 = plt.subplots()
    dff["risk_score"].plot(kind="hist", bins=20, ax=ax3)
    ax3.set_xlabel("Risk Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Risk Score Distribution")
    fig3.tight_layout()
    st.pyplot(fig3)

# ---------- Top flagged cases ----------
st.subheader("Top High-Priority Cases")
cols = [c for c in [
    "case_id","provider","payer","drg",
    "length_of_stay","readmission_30d","mismatch",
    "prov_budget_pct","prov_payer_pct","risk_score","priority_bucket"
] if c in dff.columns]
top = dff.sort_values(["priority_bucket","risk_score"], ascending=[True, False]).head(25) if "priority_bucket" in dff.columns else dff.sort_values("risk_score", ascending=False).head(25)
st.dataframe(top[cols] if cols else top.head(25))

# ---------- Download filtered data ----------
@st.cache_data
def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download filtered CSV",
    data=to_csv_bytes(dff),
    file_name="filtered_cases.csv",
    mime="text/csv",
)

# ---------- Footer ----------
st.caption(
    "Data is synthetic and for practice only. If only cases.csv is found, the app enriches it on the fly "
    "to include variance fields and a simple, explainable risk score."
)
