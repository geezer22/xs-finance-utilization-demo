
# XSOLIS Finance/Utilization Demo

A quick, self-contained repo to showcase healthcare **finance & utilization analytics** aligned with XSOLIS' mission:
- Synthetic **patient cases** across 2024–2025
- KPIs: **LOS**, **readmission rate**, **payer/provider mismatch**, **budget variance**
- Visuals: LOS trend, cost distributions, mismatch rate by provider, variance by DRG

## Repo Structure
```
xs-finance-utilization-demo/
├─ data/
│  └─ cases.csv
├─ images/
│  ├─ avg_los_by_month.png
│  ├─ cost_distribution_box.png
│  ├─ mismatch_rate_by_provider.png
│  └─ variance_by_drg.png
├─ xs_finance_utilization_demo.ipynb
├─ requirements.txt
└─ README.md
```

## How to Run
```bash
pip install -r requirements.txt
jupyter lab  # or jupyter notebook
# open xs_finance_utilization_demo.ipynb
```

## Example Outputs
![LOS](images/avg_los_by_month.png)
![Box](images/cost_distribution_box.png)
![Mismatch](images/mismatch_rate_by_provider.png)
![Variance](images/variance_by_drg.png)

## Talking Points (Interview)
- **Finance impact**: quantify provider vs budget variance and provider vs payer differences in $ terms.
- **Operational insight**: highlight high-variance DRGs; flag providers with elevated mismatch rates.
- **Scalability**: this synthetic pipeline can be wired to live feeds (claims/UR data) and pushed to a BI tool.
- **Extend**: add case-priority heuristics or a simple risk model to triage reviews.
