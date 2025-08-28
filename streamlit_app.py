# streamlit_app.py
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

# ---------- Page setup ----------
st.set_page_config(page_title="XS Finance & Utilization Demo", layout="wide")
st.title("XSOLIS-style Finance & Utilization Demo (Synthetic)")

# ---------- Config: GitHub raw fallbacks ----------
GH_BASE = "https://raw.githubusercontent.com/geezer22/xs-finance-utilization-demo/main/data"
GH_ENRICHED = f"{GH_BASE}/cases_enriched.csv"
GH_RAW      = f"{GH_BASE}/cases.csv"

# ---------- Generate synthetic data as ultimate fallback ----------
def generate_synthetic_data(n_cases=1000):
    """Generate synthetic healthcare data as ultimate fallback."""
    np.random.seed(42)
    random.seed(42)
    
    providers = ["Memorial Health", "City General", "Regional Medical", "St. Mary's", "University Hospital"]
    payers = ["Medicare", "Medicaid", "Blue Cross", "Aetna", "UnitedHealth", "Cigna", "Self-Pay"]
    drgs = ["DRG-001", "DRG-002", "DRG-003", "DRG-004", "DRG-005"]
    
    # Generate dates over the past year
    start_date = datetime.now() - timedelta(days=365)
    
    data = []
    for i in range(n_cases):
        admit_date = start_date + timedelta(days=np.random.randint(0, 365))
        los = max(1, int(np.random.lognormal(1.5, 0.8)))  # Realistic LOS distribution
        discharge_date = admit_date + timedelta(days=los)
        
        # Base costs with realistic relationships
        base_cost = np.random.lognormal(8.5, 0.7)  # ~$5k-50k range
        provider_cost = base_cost * np.random.normal(1.0, 0.15)
        payer_cost = base_cost * np.random.normal(0.85, 0.12)  # Payers typically pay less
        budget_cost = base_cost * np.random.normal(0.95, 0.10)  # Budget estimates
        
        case = {
            "case_id": f"CASE-{i+1:04d}",
            "provider": np.random.choice(providers),
            "payer": np.random.choice(payers),
            "drg": np.random.choice(drgs),
            "admit_date": admit_date,
            "discharge_date": discharge_date,
            "length_of_stay": los,
            "provider_cost": max(0, provider_cost),
            "payer_cost": max(0, payer_cost),
            "budget_cost": max(0, budget_cost),
            "readmission_30d": int(np.random.random() < 0.12),  # ~12% readmission rate
            "drg_severity": np.random.beta(2, 5),  # Skewed toward lower severity
        }
        data.append(case)
    
    return pd.DataFrame(data)

# ---------- Data loader (robust for cloud) ----------
@st.cache_data
def load_data():
    """
    Load data in this order:
      1) Local: <repo>/data/cases_enriched.csv  (preferred)
      2) Local: <repo>/data/cases.csv           (then enrich)
      3) GitHub raw URL: cases_enriched.csv
      4) GitHub raw URL: cases.csv (then enrich)
      5) Generate synthetic data (ultimate fallback)
    """
    base = Path(__file__).resolve().parent
    data_dir = base / "data"
    cwd_dir = Path.cwd() / "data"

    # Debug: show paths and what we can see
    try:
        local_files = sorted(p.name for p in data_dir.glob("*.csv")) if data_dir.exists() else []
    except Exception:
        local_files = []
    try:
        cwd_files = sorted(p.name for p in cwd_dir.glob("*.csv")) if cwd_dir.exists() else []
    except Exception:
        cwd_files = []

    st.caption(
        f"Base: {base} | CWD: {Path.cwd()}  | "
        f"data_dir exists={data_dir.exists()}, files={local_files}  | "
        f"cwd/data exists={cwd_dir.exists()}, files={cwd_files}"
    )

    # 1) Local enriched
    enriched_path = data_dir / "cases_enriched.csv"
    if enriched_path.exists():
        try:
            return pd.read_csv(enriched_path, parse_dates=["admit_date","discharge_date"])
        except Exception as e:
            st.warning(f"Error reading local enriched file: {e}")

    # 1b) Local enriched via CWD
    enriched_cwd = cwd_dir / "cases_enriched.csv"
    if enriched_cwd.exists():
        try:
            return pd.read_csv(enriched_cwd, parse_dates=["admit_date","discharge_date"])
        except Exception as e:
            st.warning(f"Error reading CWD enriched file: {e}")

    # 2) Local raw -> enrich
    raw_path = data_dir / "cases.csv"
    if raw_path.exists():
        try:
            return enrich(pd.read_csv(raw_path, parse_dates=["admit_date","discharge_date"]))
        except Exception as e:
            st.warning(f"Error reading local raw file: {e}")

    # 2b) Local raw via CWD -> enrich
    raw_cwd = cwd_dir / "cases.csv"
    if raw_cwd.exists():
        try:
            return enrich(pd.read_csv(raw_cwd, parse_dates=["admit_date","discharge_date"]))
        except Exception as e:
            st.warning(f"Error reading CWD raw file: {e}")

    # 3) GitHub raw: enriched
    try:
        df_url = pd.read_csv(GH_ENRICHED, parse_dates=["admit_date","discharge_date"])
        st.caption("✅ Loaded from GitHub raw: cases_enriched.csv")
        return df_url
    except Exception as e:
        st.caption(f"⚠️ Could not load from GitHub enriched: {e}")

    # 4) GitHub raw: raw -> enrich
    try:
        df_url = pd.read_csv(GH_RAW, parse_dates=["admit_date","discharge_date"])
        st.caption("✅ Loaded from GitHub raw: cases.csv (auto-enriched)")
        return enrich(df_url)
    except Exception as e:
        st.caption(f"⚠️ Could not load from GitHub raw: {e}")

    # 5) Ultimate fallback: generate synthetic data
    st.info("🔧 No data files found. Generating synthetic data for demo purposes.")
    return enrich(generate_synthetic_data())

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal enrichment so the app works even if only cases.csv exists.

    Creates:
      - prov_budget_pct / prov_payer_pct (+ clipped versions)
      - mismatch flag if not present
      - robust-scaled LOS (los_norm)
      - simple risk_score (0–100) + priority_bucket
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

# ---------- Load data ----------
df = load_data()

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
c2.metric("Avg LOS (days)", f"{los_mean:.1f}" if np.isfinite(los_mean) else "—")

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
    
    if len(los_by_month) > 0:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        los_by_month.plot(ax=ax1, marker='o')
        ax1.set_title("Average Length of Stay by Month")
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Avg LOS (days)")
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        st.pyplot(fig1)
    else:
        st.info("No data available for LOS trend chart")

# ---------- Chart 2: Mismatch rate by provider ----------
st.subheader("Mismatch Rate by Provider")
if "mismatch" in dff.columns and "provider" in dff.columns:
    mm = dff.groupby("provider")["mismatch"].mean().sort_values()
elif {"provider", "prov_payer_pct"} <= set(dff.columns):
    mm = dff.groupby("provider")["prov_payer_pct"].apply(lambda s: (s.abs() > 0.05).mean()).sort_values()
else:
    mm = pd.Series(dtype=float)

if len(mm) > 0:
    fig2, ax2 = plt.subplots(figsize=(10, max(4, len(mm) * 0.3)))
    mm.plot(kind="barh", ax=ax2, color='lightcoral')
    ax2.set_xlabel("Mismatch Rate")
    ax2.set_ylabel("Provider")
    ax2.set_title("Mismatch Rate by Provider")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2)
else:
    st.info("No data available for mismatch rate chart")

# ---------- Chart 3: Risk distribution ----------
st.subheader("Risk Score Distribution")
if "risk_score" in dff.columns and len(dff) > 0:
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    dff["risk_score"].plot(kind="hist", bins=20, ax=ax3, color='skyblue', alpha=0.7)
    ax3.set_xlabel("Risk Score")
    ax3.set_ylabel("Count")
    ax3.set_title("Risk Score Distribution")
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    st.pyplot(fig3)
else:
    st.info("No data available for risk score distribution")

# ---------- Top flagged cases ----------
st.subheader("Top High-Priority Cases")
cols = [c for c in [
    "case_id","provider","payer","drg",
    "length_of_stay","readmission_30d","mismatch",
    "prov_budget_pct","prov_payer_pct","risk_score","priority_bucket"
] if c in dff.columns]

if len(dff) > 0:
    top = (
        dff.sort_values(["priority_bucket","risk_score"], ascending=[True, False]).head(25)
        if "priority_bucket" in dff.columns
        else dff.sort_values("risk_score", ascending=False).head(25)
    )
    
    # Format percentage columns for display
    display_df = top[cols] if cols else top.head(25)
    if "prov_budget_pct" in display_df.columns:
        display_df = display_df.copy()
        display_df["prov_budget_pct"] = display_df["prov_budget_pct"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")
    if "prov_payer_pct" in display_df.columns:
        display_df["prov_payer_pct"] = display_df["prov_payer_pct"].apply(lambda x: f"{x:.1%}" if pd.notnull(x) else "—")
    
    st.dataframe(display_df, use_container_width=True)
else:
    st.info("No cases match the current filters")

# ---------- Download filtered data ----------
@st.cache_data
def to_csv_bytes(df_in: pd.DataFrame) -> bytes:
    return df_in.to_csv(index=False).encode("utf-8")

if len(dff) > 0:
    st.download_button(
        "📥 Download filtered CSV",
        data=to_csv_bytes(dff),
        file_name="filtered_cases.csv",
        mime="text/csv",
    )

# ---------- Footer ----------
st.markdown("---")
st.caption(
    "🔬 **Demo Note:** Data is synthetic and for demonstration purposes only. "
    "The app automatically generates realistic healthcare data if no source files are found. "
    "Risk scores are calculated using a simple model combining LOS, cost variances, and readmission history."
)
