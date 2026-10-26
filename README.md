# Job Acceptance Prediction System

Predicts whether a candidate will accept or reject a job offer using machine learning.
Domain: HR Analytics & Predictive Modeling.

## Project files

| File | What it does |
|---|---|
| `Dataset.csv` | raw dataset (51,500 candidate records) |
| `01_Data_Cleaning.ipynb` | removes duplicates, fixes inconsistent labels, handles missing values -> `Cleaned_Dataset.csv` |
| `02_Exploratory_Data_Analysis.ipynb` | charts and insights -> saved in `figures/` |
| `03_Feature_Engineering.ipynb` | creates experience category, academic band, skills level, interview category, CTC gap -> `Final_Dataset.csv` |
| `04_SQL_Database_Connection.ipynb` | stores the cleaned data in a database (MySQL, falls back to SQLite `job_acceptance.db`) |
| `05_Model_Building.ipynb` | trains and compares 3 models, saves the best one as `model.pkl` |
| `app.py` | Streamlit dashboard with KPIs, EDA charts and a prediction page |
| `model.pkl` | trained Random Forest model + scaler + column names |

## How to run

```bash
pip install -r requirements.txt
```

1. Run the notebooks in order: `01` -> `02` -> `03` -> `04` -> `05`
   (in `04_SQL_Database_Connection.ipynb` change the `password` variable to your MySQL root password)
2. Start the dashboard:

```bash
streamlit run app.py
```

## Results (test set)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| **Random Forest (selected)** | **88.14%** | **81.13%** | **0.9518** |
| Logistic Regression | 85.18% | 77.93% | 0.9371 |
| Decision Tree | 81.25% | 74.24% | 0.9135 |

* Overall placement rate: 30.3% (data is imbalanced, handled with stratified split + class_weight)
* Strongest predictors: technical score, skills match %, CTC expectation gap, years of experience
* Acceptance rises steeply with interview score: ~0% below 50, ~76% above 85
