# XSOLIS Finance/Utilization Demo

This is a practice project I built to connect my finance/analytics background with healthcare-style data.  
All data here is synthetic — created only for practice — but the metrics and workflow mirror common healthcare ops/finance reporting.

---

## What’s Inside
- **Synthetic patient cases** (2024–2025) with admit/discharge dates, DRG, costs, and readmission flags  
- **KPIs**  
  - Average Length of Stay (LOS)  
  - Readmission rate  
  - Provider vs Payer mismatches  
  - Budget vs Actual variance  
- **Notebook** with analysis, charts, and KPI logic  
- **Streamlit app** for filtering by provider/payer/DRG and exporting CSVs  
- **Visuals**: LOS trend, cost distributions, mismatch rate, DRG variance, and a budget → provider → payer waterfall

---

## Repo Structure
```
xs-finance-utilization-demo/
├─ data/
│  ├─ cases.csv
│  └─ cases_enriched.csv
├─ images/
│  ├─ avg_los_by_month.png
│  ├─ cost_distribution_box.png
│  ├─ mismatch_rate_by_provider.png
│  ├─ variance_by_drg.png
│  └─ waterfall_budget_provider_payer.png
├─ xs_finance_utilization_demo.ipynb
├─ streamlit_app.py
├─ config.json
├─ requirements.txt
└─ README.md
```

---

## How to Run

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

### Notebook
```bash
pip install -r requirements.txt
jupyter lab   # or: jupyter notebook
```
Open `xs_finance_utilization_demo.ipynb` and run all cells.

---

## Example Outputs

Average Length of Stay by Month  
![LOS](images/avg_los_by_month.png)

Cost Distribution  
![Cost](images/cost_distribution_box.png)

Mismatch Rate by Provider  
![Mismatch](images/mismatch_rate_by_provider.png)

Variance by DRG  
![Variance](images/variance_by_drg.png)

Budget → Provider → Payer Waterfall  
![Waterfall](images/waterfall_budget_provider_payer.png)

---

## Notes
- All data is fake and only for practice.  
- Goal: demonstrate how I approach **KPI design, reconciliation, and reporting** across finance and utilization metrics.  
