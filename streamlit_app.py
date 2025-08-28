
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="XS Finance & Utilization Demo", layout="wide")
st.title("XSOLIS-style Finance & Utilization Demo (Synthetic)")

@st.cache_data
def load():
    # Resolve path relative to this file, not the working directory
    base = Path(__file__).parent.resolve()
    csv_path = base / "data" / "cases_enriched.csv"
    if not csv_path.exists():
        # Fallback to CWD just in case
        csv_path = Path("data") / "cases_enriched.csv"
    df = pd.read_csv(csv_path, parse_dates=["admit_date", "discharge_date"])
    return df

df = load()

# Sidebar filters
providers = ["All"] + sorted(df["provider"].unique().tolist())
payers = ["All"] + sorted(df["payer"].unique().tolist())
drgs = ["All"] + sorted(df["drg"].unique().tolist())

p_sel = st.sidebar.selectbox("Provider", providers, index=0)
pay_sel = st.sidebar.selectbox("Payer", payers, index=0)
drg_sel = st.sidebar.selectbox("DRG", drgs, index=0)

mask = np.ones(len(df), dtype=bool)
if p_sel != "All": mask &= (df["provider"] == p_sel)
if pay_sel != "All": mask &= (df["payer"] == pay_sel)
if drg_sel != "All": mask &= (df["drg"] == drg_sel)
dff = df.loc[mask].copy()

# KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Cases", len(dff))
c2.metric("Avg LOS (days)", f"{dff['length_of_stay'].mean():.2f}")
c3.metric("Readmit Rate", f"{dff['readmission_30d'].mean():.1%}")
c4.metric("Mismatch Rate", f"{dff['mismatch'].mean():.1%}")
c5.metric("Avg Risk Score", f"{dff['risk_score'].mean():.1f}")

# Chart 1: LOS trend
dff["admit_month"] = dff["admit_date"].dt.to_period("M").dt.to_timestamp()
los_by_month = dff.groupby("admit_month")["length_of_stay"].mean()

fig1 = plt.figure()
los_by_month.plot(kind="line", title="Average Length of Stay by Month")
plt.xlabel("Month"); plt.ylabel("Avg LOS (days)"); plt.tight_layout()
st.pyplot(fig1)

# Chart 2: Mismatch rate by provider
mismatch_rate = dff.groupby("provider")["mismatch"].mean().sort_values()
fig2 = plt.figure()
mismatch_rate.plot(kind="barh", title="Mismatch Rate by Provider")
plt.xlabel("Mismatch Rate"); plt.tight_layout()
st.pyplot(fig2)

# Chart 3: Risk distribution
fig3 = plt.figure()
dff["risk_score"].plot(kind="hist", bins=20, title="Risk Score Distribution")
plt.xlabel("Risk Score"); plt.tight_layout()
st.pyplot(fig3)

# Top flagged cases
st.subheader("Top High-Priority Cases")
top = dff.sort_values(["priority_bucket","risk_score"], ascending=[True, False]).head(25)
st.dataframe(top[["case_id","provider","payer","drg","length_of_stay","readmission_30d","mismatch","prov_budget_pct","prov_payer_pct","risk_score","priority_bucket"]])

# Download filtered data
st.download_button("Download filtered CSV", data=dff.to_csv(index=False), file_name="filtered_cases.csv", mime="text/csv")

st.caption("Charts use matplotlib only. Risk scoring uses monotonic, weighted features with robust scaling and sigmoid calibration.")
