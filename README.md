# XSOLIS Finance/Utilization Demo

>  **Note on the Live Demo**  
> This project includes a Streamlit app for interactive exploration. The app is deployed on Streamlit Cloud,  
> but due to file path and container quirks it may occasionally fail to load the data.  
> 
> The **code, notebook, and CSVs in this repo are complete and fully runnable locally** with:
> ```bash
> pip install -r requirements.txt
> streamlit run streamlit_app.py
> ```
> 
> I’ve also included screenshots of the outputs in the `images/` folder and the notebook  
> (`xs_finance_utilization_demo.ipynb`) so you can view the analysis without relying on the live demo.

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
xs-finance-utilization-demo/
├─ data/
│ ├─ cases.csv
│ └─ cases_enriched.csv
├─ images/
│ ├─ avg_los_by_month.png
│ ├─ cost_distribution_box.png
│ ├─ mismatch_rate_by_provider.png
│ ├─ variance_by_drg.png
│ └─ waterfall_budget_provider_payer.png
├─ xs_finance_utilization_demo.ipynb
├─ streamlit_app.py
├─ config.json
├─ requirements.txt
└─ README.md

yaml
Copy code

---

## How to Run

### Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
Then open the local URL Streamlit prints (usually http://localhost:8501).

Notebook
bash
Copy code
pip install -r requirements.txt
jupyter lab   # or: jupyter notebook
Open xs_finance_utilization_demo.ipynb and run all cells.

Example Outputs
Average Length of Stay by Month

Cost Distribution

Mismatch Rate by Provider

Variance by DRG

Budget → Provider → Payer Waterfall

Talking Points (Interview)
Finance impact: quantify budget variance and mismatches between providers/payers in $ terms

Ops insights: highlight higher-variance DRGs and providers with elevated readmission %

Scalability: pipeline could feed BI dashboards or a real-time Streamlit app with hospital feeds

Extendability: add cost priority heuristics or a simple risk model to flag trends

Notes
All data is fake and only for practice.

Goal: demonstrate how I approach KPI design, reconciliation, and reporting across finance and utilization metrics.
