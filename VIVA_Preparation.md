# Viva Preparation - Job Acceptance Prediction System

This guide is written only for this project. Every number below comes from the
notebooks in this folder, so you can defend it in the viva.

How to use it:

1. Read sections 1-7 once (about 40 minutes).
2. Practise the questions in section 8 with the file closed.
3. Memorise section 9 (numbers) and section 10 (demo script) on the last day.

---

## 1. Project in one minute (elevator pitch)

> "My project is a Job Acceptance Prediction System in the HR analytics domain.
> Companies lose time and money when a candidate rejects an offer at the last
> moment. I built a machine learning model that takes a candidate's profile -
> academics, interview scores, skills match, experience, CTC details, company
> tier and so on - and predicts the probability that the candidate will accept
> the offer. I cleaned a 51,500-row dataset down to 50,000 records, did EDA,
> created five new HR-friendly features, stored the data in a database, trained
> and compared Logistic Regression, Decision Tree and Random Forest, and Random
> Forest gave the best ROC-AUC of 0.9518 with 88.14% accuracy. Finally I built a
> Streamlit dashboard where an HR user can see KPIs, EDA charts and predict a
> single candidate's acceptance probability."

That is the whole viva in 45 seconds. Everything else is detail on top of it.

---

## 2. Problem statement and objective

**Business problem.** An offer rejection late in the hiring process wastes
interview panels, recruiter time and joining plans. HR wants to know in advance
which candidates are likely to accept.

**Technical problem.** Binary classification: given a candidate profile `X`,
predict `y = 1` (accept) or `y = 0` (reject), and also output the probability.

**Inputs.** 25 raw candidate attributes (demographics, academics, interview
scores, skills, experience, CTC, company/job details).

**Output.** Acceptance probability (0-100%) and a class label at the 0.5
threshold.

**Use in HR.** Prioritise candidates who are likely to accept, keep backup
candidates ready for high-risk offers, and understand which factors drive
acceptance (for example interview score and skills match).

**Type of ML.** Supervised learning -> classification (not regression, because
the target is a category, not a continuous number).

---

## 3. Dataset

| Item | Value |
|---|---|
| Raw file | `Dataset.csv` - 51,500 rows |
| After cleaning | `Cleaned_Dataset.csv` - 50,000 rows, 0 missing values |
| Original features | 25 |
| Engineered features | 5 (notebook 03) |
| Final columns | 31 (25 + 5 + target `status`) |
| Target | `status`: `Placed` = accepted the offer (1), `Not Placed` = did not accept (0) |
| Class balance | 15,130 Placed (30.26%) vs 34,870 Not Placed (69.74%) |
| Train/test | 40,000 (80%) / 10,000 (20%), stratified |

The classes are imbalanced (roughly 30:70). This matters because a dumb model
that always predicts "Not Placed" already gets 69.74% accuracy - so accuracy
alone is not a good metric here.

**Feature groups (be ready to name a few):**

| Group | Columns |
|---|---|
| Personal | `age_years`, `gender` |
| Academics | `ssc_percentage`, `hsc_percentage`, `degree_percentage`, `degree_specialization` |
| Interview | `technical_score`, `aptitude_score`, `communication_score` |
| Skills | `skills_match_percentage`, `certifications_count`, `internship_experience` |
| Experience | `years_of_experience`, `relevant_experience`, `career_switch_willingness`, `employment_gap_months`, `layoff_history` |
| CTC | `previous_ctc_lpa`, `expected_ctc_lpa` |
| Job/company | `company_tier`, `job_role_match`, `competition_level`, `bond_requirement`, `notice_period_days`, `relocation_willingness` |
| Target | `status` |

If asked "is this real company data?", answer honestly: it is a public /
educational HR analytics dataset used for academic purposes; the pipeline is the
same one you would use on real data after privacy checks.

---

## 4. End-to-end pipeline

```
Dataset.csv (51,500)
   |
   v
01_Data_Cleaning.ipynb  ->  Cleaned_Dataset.csv (50,000, 0 missing)
   |
   v
02_Exploratory_Data_Analysis.ipynb  ->  6 figures + insights
   |
   v
03_Feature_Engineering.ipynb  ->  Final_Dataset.csv (31 columns)
   |
   v
04_SQL_Database_Connection.ipynb  ->  MySQL (or SQLite fallback) + summary table
   |
   v
05_Model_Building.ipynb  ->  model.pkl (RF + scaler + column list)
   |
   v
app.py (Streamlit)  ->  Overview | EDA | Predict Candidate
```

Run order: 01 -> 02 -> 03 -> 04 -> 05 -> `streamlit run app.py`.

---

## 5. Notebook-by-notebook explanation

### 5.1 Notebook 01 - Data cleaning

What the code does, in order:

1. `pd.read_csv("Dataset.csv")` and checks `shape`, `info()`, `isnull().sum()`,
   `describe()`.
2. **Removes duplicates in two stages.**
   - First `drop_duplicates()` removes 1,376 exact duplicates.
   - Then labels are standardised (`male`/`female` -> `Male`/`Female`,
     `internship_experience` stripped and title-cased, tier columns stripped).
     After standardisation another 124 duplicate rows appear, because the same
     record was written with different letter cases/spaces. They are removed
     too.
   - Total removed = 1,500 -> exactly 50,000 rows.
3. **Fixes inconsistent categories** so `get_dummies` does not create extra
   columns for `male`, `Male`, `MALE`.
4. **Fills missing values**:
   - `ssc_percentage`, `hsc_percentage`, `notice_period_days` -> **median**.
   - `employment_gap_months` -> **0** (no gap recorded = no gap).
   - `career_switch_willingness`, `relevant_experience`, `job_role_match`,
     `layoff_history`, `relocation_willingness` -> **mode** (most frequent
     category).
5. **Fresher consistency rule**: if `years_of_experience == 0`, set
   `previous_ctc_lpa = 0`. A fresher cannot have a previous salary.
6. Saves `Cleaned_Dataset.csv` and prints the placement rate (30.26%).

Why median and not mean? These columns are bounded percentages/days and can be
skewed; the median is robust to extreme values. Why not drop missing rows?
Missing values are limited to 9 columns but reach about 16% in some categorical
columns; dropping every incomplete row would waste thousands of valid records.
Why no outlier removal? The scores have valid domain bounds (0-100, CTC >= 0),
and tree-based models are robust to outliers. Outlier removal could delete
genuine high performers.

If the examiner asks "why check duplicates twice?" - because duplicates can hide
behind inconsistent text labels; cleaning the labels revealed 124 more.

### 5.2 Notebook 02 - Exploratory Data Analysis

Purpose: understand distributions and relationships before modelling, and get
evidence for feature choices.

Six figures saved in `figures/`:

1. `1_target_distribution.png` - count plot: 34,870 Not Placed vs 15,130 Placed.
   Shows the imbalance.
2. `2_acceptance_by_category.png` - acceptance rate by company tier,
   competition level, job role match and internship. Key finding: company tier
   is almost flat (Tier 1 29.9%, Tier 2 30.4%, Tier 3 30.5%), while job role
   match matters (Matched 34.7% vs Not Matched 15.7%) and internship matters
   (Yes 37.0% vs No 24.1%).
3. `3_interview_vs_acceptance.png` - box plot + binned line. This is the
   strongest pattern: acceptance is about 0% below an average interview score of
   50, and rises steadily to about 77% in the top band (87+). Technical score has
   the highest correlation with placement (0.404).
4. `4_skills_match_impact.png` - count + acceptance rate by skills band:
   Low (<60) 11.1%, Medium (60-85) 30.8%, High (>85) 41.4%.
5. `5_experience_vs_placement.png` - Fresher 13.8%, Junior 29.2%, Senior 53.7%.
6. `6_correlation_heatmap.png` - Pearson correlation of numeric columns only,
   upper triangle masked. Used to spot related variables and check for very high
   multicollinearity.

Important insight: correlation does not prove causation. The heatmap and rates
tell us which variables are *associated* with acceptance; the model then learns
how to combine them.

### 5.3 Notebook 03 - Feature engineering

Five new features (domain-driven binning):

| New feature | Definition | Why |
|---|---|---|
| `experience_category` | 0 = Fresher, 1-2 = Junior, 3+ = Senior | HR thinks in bands, not exact years; captures non-linearity |
| `academic_performance_band` | mean of SSC/HSC/degree: Low <60, Medium 60-75, High >75 | One overall academic signal instead of three noisy ones |
| `skills_match_level` | Low <60, Medium 60-85, High >85 | EDA showed a clear step pattern |
| `interview_performance_category` | mean of three interview scores: Weak <55, Average 55-70, Strong >70 | Converts scores into HR-meaningful levels |
| `ctc_expectation_gap_lpa` | expected CTC - previous CTC | Measures how big a jump the candidate expects; salary mismatch influences acceptance |

All bin edges come from the distribution/EDA, not from the target, so there is
no label leakage. The original numeric columns are kept, so binning adds
information rather than replacing detail. `.astype(str)` is used so `pd.cut`
results are treated as categories by `get_dummies`.

Output: `Final_Dataset.csv` (31 columns).

### 5.4 Notebook 04 - SQL database connection

- Uses **SQLAlchemy** `create_engine` with PyMySQL for MySQL
  (`mysql+pymysql://root:password@localhost/job_acceptance_db`).
- Creates the database if it does not exist, then writes two tables:
  - `candidates` - all 50,000 cleaned records (one row per candidate).
  - `acceptance_summary_by_tier` - 3-row summary with total candidates,
    accepted count and acceptance rate per company tier.
- If MySQL is unavailable (wrong password / server down), the code catches the
  exception and falls back to **SQLite** (`job_acceptance.db`), so the notebook
  still runs on any machine.
- `to_sql(..., if_exists="replace", chunksize=1000)` writes in batches.
- Ends with `pd.read_sql("SELECT COUNT(*) ...")` to verify the data is really in
  the database.

Why a database instead of only CSV? Persistent central storage, SQL querying for
reporting, multiple users/tools can read it, and it avoids re-processing the
CSV every time.

Note: in a submitted project the password should be a placeholder (as here) or
read from an environment variable - never a real password.

### 5.5 Notebook 05 - Model building

Step by step:

1. `target = (status == "Placed")` -> 1/0.
2. `X = df.drop(columns=["status", "target"])`, `y = target`.
3. **Leakage check.** The label is copied into a numeric `target` column
   (Placed = 1). Both `status` (the original label) and `target` (its numeric
   copy) are dropped from `X`; otherwise the model sees the answer and reports
   100% accuracy. This defensive drop matters: an intermediate CSV during
   development once still contained `target`, and every model scored AUC 1.0 -
   a warning sign of leakage, not a good model. After removing it, the honest
   result was around 88%.
4. **Encoding**: `pd.get_dummies(X, drop_first=True)` -> 30 features become 39
   encoded columns. `drop_first=True` avoids the dummy variable trap (perfect
   multicollinearity when one category is implied by the others).
5. **Split**: `train_test_split(test_size=0.2, random_state=42, stratify=y)` ->
   40,000 train / 10,000 test, same 30.3% positive rate in both.
6. **Scaling**: `StandardScaler` on the original numeric columns only. It is
   `fit_transform` on train and only `transform` on test - fitting on the test
   set would leak test statistics. Trees do not need scaling, but Logistic
   Regression does; keeping one scaled matrix makes the comparison fair and the
   app reuses the same scaler.
7. **Three models**, all with `class_weight="balanced"`:
   - Logistic Regression - linear baseline, `max_iter=2000`.
   - Decision Tree - non-linear, `max_depth=8` to control overfitting.
   - Random Forest - 200 trees, `n_jobs=-1`.
8. Evaluates with accuracy, precision, recall, F1 and ROC-AUC, prints a
   confusion matrix for Random Forest, plots the top 15 feature importances, and
   saves `model.pkl`.

Why Random Forest won: it combines many decorrelated trees (bagging), captures
non-linear interactions such as "high technical score + low competition", and
averaging reduces the variance that hurts a single decision tree.

What is saved in `model.pkl`:

```python
joblib.dump({"model": best_estimator,
             "scaler": scaler,
             "columns": list(X.columns),
             "num_cols": num_cols}, "model.pkl")
```

All four objects are needed so the Streamlit app can reproduce the exact
training preprocessing for one new candidate.

---

## 6. Evaluation - understand every number

### 6.1 Confusion matrix (Random Forest, 10,000 test rows)

|  | Predicted Not Placed | Predicted Placed |
|---|---|---|
| **Actual Not Placed** | TN = 6,265 | FP = 709 |
| **Actual Placed** | FN = 477 | TP = 2,549 |

In HR language:

- TP = correctly predicted an acceptance.
- TN = correctly predicted a rejection.
- FP = model said "will accept", but the candidate rejected -> wasted planning,
  need a backup.
- FN = model said "will reject", but the candidate accepted -> missed
  opportunity / unnecessary backup.

### 6.2 Formulas and values for Random Forest

- Accuracy = (TP + TN) / total = (2549 + 6265) / 10000 = **88.14%**
- Precision = TP / (TP + FP) = 2549 / 3258 = **78.24%**
- Recall (sensitivity) = TP / (TP + FN) = 2549 / 3026 = **84.24%**
- F1 = 2 x (P x R) / (P + R) = **81.13%**
- ROC-AUC = **0.9518**

### 6.3 Model comparison (test set)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest (selected)** | **88.14%** | **78.24%** | **84.24%** | **81.13%** | **0.9518** |
| Logistic Regression | 85.18% | 70.92% | 86.48% | 77.93% | 0.9371 |
| Decision Tree | 81.25% | 63.53% | 89.29% | 74.24% | 0.9135 |

Selection rule: highest ROC-AUC. Random Forest also has the best accuracy and
F1, so it is the clear choice.

### 6.4 Why these metrics

- **Accuracy** alone is misleading with 30:70 imbalance - a constant "Not
  Placed" predictor gets 69.74%. Our 88.14% is about 18.4 points above that
  baseline, but precision/recall tell the real story.
- **Precision** answers: when the model says "will accept", how often is it
  right? Important if HR acts on the prediction (backup planning).
- **Recall** answers: of all candidates who actually accept, how many did we
  catch? Important if missing an acceptor is expensive.
- **F1** balances precision and recall in one number.
- **ROC-AUC** measures ranking quality across all thresholds; 0.5 is random,
  1.0 is perfect. It is threshold-independent, which is why it is used to pick
  the model.
- **Threshold**: predictions use the default 0.5. Lowering it (e.g. 0.4) raises
  recall and lowers precision; raising it does the opposite. The business cost
  of FP vs FN decides the threshold.

### 6.5 Feature importance (Random Forest, top 10)

| Feature | Importance |
|---|---|
| technical_score | 0.2456 |
| skills_match_percentage | 0.0755 |
| ctc_expectation_gap_lpa | 0.0750 |
| years_of_experience | 0.0518 |
| previous_ctc_lpa | 0.0462 |
| expected_ctc_lpa | 0.0428 |
| job_role_match_Not Matched | 0.0403 |
| communication_score | 0.0379 |
| interview_performance_category_Strong | 0.0320 |
| experience_category_Senior | 0.0315 |

Random Forest importance is **mean decrease in impurity** (Gini importance),
averaged over all trees and normalized. Technical score dominates, which matches
the EDA. Importance shows how much a feature helps the model split; it is not a
causal effect.

---

## 7. Streamlit dashboard (`app.py`)

What happens when the app starts:

1. Reads `Cleaned_Dataset.csv` and `model.pkl`.
2. Defines `create_features()` - the same 5 features as notebook 03.
3. Defines `predict_acceptance(raw_df)`:
   - `create_features`
   - `pd.get_dummies(drop_first=True)`
   - `reindex(columns=model_columns, fill_value=0)` - forces exactly the 39
     training columns; any category absent for one candidate becomes 0.
   - `scaler.transform(num_cols)` - same scaler fitted on the training data.
   - `model.predict_proba(...)[:, 1]` - probability of class 1 (accept).
4. Batch-predicts all 50,000 candidates once to compute the High-Risk KPI
   (predicted probability < 50%).
5. Sidebar navigation with three pages:
   - **Dashboard Overview** - 7 KPI cards, acceptance by tier, status split,
     interview score curve, skills match curve.
   - **EDA Insights** - model comparison table + all 7 saved figures.
   - **Predict Candidate** - a 3-column form, probability gauge and
     accept/reject message.

Why reindex? Because a single candidate does not contain every category, so
`get_dummies` would create fewer columns than the model expects and prediction
would crash. Reindex restores the exact training schema.

Why the scaler is saved? The model was trained on scaled values; a new input must
be scaled with the **same mean and standard deviation**, so the fitted scaler
must be persisted.

Run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app is a local dashboard (localhost:8501). It is not deployed publicly, and
it has no login - a limitation for production.

---

## 8. Viva questions and answers

### A. Project and data

**Q1. What is your project about?**
A machine learning system that predicts whether a candidate will accept a job
offer, with an interactive Streamlit dashboard for HR.

**Q2. Why is this a classification problem and not regression?**
The output is a class - accept (1) or reject (0) - not a continuous quantity.
We also output a probability, but the task itself is classification.

**Q3. Supervised or unsupervised?**
Supervised, because the training data has labelled examples (`status`).

**Q4. What is the target variable and how did you encode it?**
`status`; `Placed` -> 1 (accepted), `Not Placed` -> 0 (rejected).

**Q5. How many records and features?**
51,500 raw -> 50,000 after removing 1,500 duplicates; 25 original features + 5
engineered = 30 input features, 39 after one-hot encoding.

**Q6. What is the class distribution and why does it matter?**
30.26% Placed vs 69.74% Not Placed. It is imbalanced, so accuracy alone is
misleading and we use stratified splitting, `class_weight="balanced"` and
F1/ROC-AUC.

**Q7. Where did the data come from?**
A public HR analytics dataset used for academic work. It is not private company
data; the same pipeline would apply to real data after anonymisation.

**Q8. What is the business value?**
Better offer planning: focus effort on likely acceptors, prepare backups for
high-risk candidates, and understand drivers such as interview score and skills
match.

### B. Cleaning and EDA

**Q9. Why remove duplicates?**
Duplicates bias the model toward repeated patterns and can put the same record in
both train and test, inflating the score. We removed 1,376 exact duplicates plus
124 revealed after label standardisation.

**Q10. Why did more duplicates appear after cleaning labels?**
The same candidate was stored as `male`/`Male` or with trailing spaces in a tier
column; once normalised, the rows became identical.

**Q11. Why median imputation and not mean?**
Median is robust to skew and outliers; mean would be pulled by extreme values.
For gaps we used 0, and for categoricals we used the mode.

**Q12. Why not delete rows with missing values?**
Missing values are spread over 9 columns, and the categorical ones are missing in
about 16% of rows. Dropping every incomplete row would remove thousands of
otherwise valid candidates, so median/mode/0 imputation keeps the sample size.

**Q13. Why set previous CTC to 0 for freshers?**
Consistency: a candidate with zero experience cannot have a previous salary. It
also prevents a fake salary gap feature.

**Q14. Did you remove outliers?**
No. Scores are bounded by domain (0-100), CTC cannot be negative, and tree models
are robust. Removing them could delete genuine top performers.

**Q15. What was your most important EDA insight?**
Interview performance is the strongest signal: acceptance is about 0% below an
average score of 50 and rises to about 77% in the top band. Skills match,
experience and role match also show clear patterns.

**Q16. Was company tier important?**
Barely: acceptance is about 30% in all three tiers. Not every column has to be
strongly predictive; the model can still use tier in combination with other
features. This is a useful "negative finding" from EDA.

**Q17. What does the correlation heatmap tell you?**
It shows pairwise linear relationships between numeric variables. Technical
score correlates 0.40 with placement, skills match 0.20, CTC gap -0.28. It also
helps spot multicollinearity. Correlation does not prove causation.

**Q18. Why is CTC gap negatively correlated with acceptance?**
Larger gaps are common for freshers (previous CTC = 0) and for candidates with
high expectations; those groups accept less often. It is a proxy for other
factors, not a causal salary effect.

### C. Feature engineering and preprocessing

**Q19. Why create new features?**
To encode HR-meaningful bands and interactions (experience level, academic band,
skills level, interview category, CTC gap) that raw numbers do not express
directly, and to capture non-linear effects.

**Q20. Doesn't binning lose information?**
Yes, some. That is why the original numeric columns are kept as well; the bands
add a different view rather than replacing detail.

**Q21. What is one-hot encoding?**
Converting a categorical column into binary 0/1 columns, one per category, so
the model can use categories numerically.

**Q22. Why `drop_first=True`?**
To avoid the dummy variable trap: with all categories plus an intercept, the
columns are perfectly collinear. Dropping the first category makes the remaining
dummies independent.

**Q23. Why StandardScaler?**
It transforms each numeric column to mean 0 and standard deviation 1:
`z = (x - mean) / std`. Logistic Regression is sensitive to feature scale;
trees are not, but scaling everything keeps one consistent pipeline.

**Q24. Why fit the scaler only on the training set?**
`fit_transform` on train learns mean/std; `transform` on test applies them. If we
fit on all data, the test set's statistics leak into training and the evaluation
becomes optimistic.

**Q25. Why `stratify=y`?**
It keeps the 30:70 class ratio in both train and test, so the test set
represents the real imbalance.

**Q26. What is `random_state=42`?**
A fixed seed for the random operations (split, tree randomness), so results are
reproducible.

**Q27. How do you preprocess a new candidate in the app?**
Same five engineered features -> `get_dummies` -> `reindex` to the 39 training
columns -> scale the numeric columns with the saved scaler -> `predict_proba`.

**Q28. What if a new candidate has a category the model has never seen?**
`get_dummies` creates an extra column, `reindex(columns=model_columns)` drops it,
and the row is scored with the known columns. The unseen category is treated as
the reference category, so prediction still works (though its effect is unknown).

### D. Models

**Q29. Explain Logistic Regression in one line.**
A linear model that applies the sigmoid function to a weighted sum of features
and outputs a probability between 0 and 1; trained by minimising log loss.

**Q30. Explain a Decision Tree.**
It recursively splits data on the feature/threshold that most reduces impurity
(Gini or entropy), creating a tree of if/else rules. Deep trees overfit.

**Q31. What does `max_depth=8` do?**
Limits tree depth, reducing overfitting and keeping the model interpretable.

**Q32. Explain Random Forest.**
An ensemble of decision trees trained on bootstrap samples of the data, using a
random subset of features at each split; predictions are aggregated by majority
vote (or averaged probability). It reduces variance and usually beats a single
tree.

**Q33. What is bagging?**
Bootstrap aggregating: train many models on random samples with replacement and
average their predictions. Random Forest is bagging plus random feature
selection.

**Q34. Why did Random Forest perform best?**
It captures non-linear interactions, is robust to noise and outliers, and
averaging many trees lowers variance. It achieved the highest AUC (0.9518),
accuracy (88.14%) and F1 (81.13%).

**Q35. What is `class_weight="balanced"`?**
It gives each class a weight inversely proportional to its frequency, so the
minority class (Placed) contributes more to the loss and the model does not
ignore it.

**Q36. Why is Logistic Regression recall high (86.48%) but precision low
(70.92%)?**
Because balanced class weights and a linear boundary make it flag more
candidates as likely acceptors: it catches more true acceptors (high recall) at
the cost of more false positives (lower precision).

**Q37. Why does Decision Tree have the highest recall but lowest accuracy?**
`max_depth=8` with balanced weights produces many positive predictions, so it
catches most acceptors but also many non-acceptors; its overall accuracy and
precision are lower.

**Q38. Did you tune hyperparameters?**
No. I used sensible fixed values (200 trees, depth 8, `max_iter=2000`). A next
step is `GridSearchCV` / `RandomizedSearchCV` with cross-validation.

**Q39. Why no cross-validation?**
For this project I used a single stratified 80/20 hold-out to keep the workflow
simple. Cross-validation would give a more stable estimate; it is listed as
future work.

**Q40. What is overfitting and how did you reduce it?**
Overfitting is fitting training noise, giving high train but poor test
performance. We reduce it with a hold-out test set, `max_depth`, ensemble
averaging, and by not using the leaking `target` column.

**Q41. What was the target leakage issue?**
The label `status` was copied into a numeric `target` column. During development,
an intermediate CSV still contained `target`, and every model scored 100%
accuracy / AUC 1.0. I detected the "too perfect" result, removed the label column
from the features, and got honest scores around 88% and AUC 0.95. Lesson: a
perfect score usually means the answer leaked into `X`, not that the model is
perfect.

**Q42. What is feature importance in Random Forest?**
The average decrease in node impurity produced by a feature across all trees,
normalized to sum to 1. Technical score was 0.246, by far the strongest.

### E. Metrics

**Q43. Define accuracy, precision, recall and F1.**
Accuracy = (TP+TN)/total. Precision = TP/(TP+FP). Recall = TP/(TP+FN).
F1 = 2PR/(P+R).

**Q44. Interpret your Random Forest confusion matrix.**
6,265 true negatives, 709 false positives, 477 false negatives, 2,549 true
positives on 10,000 test records.

**Q45. Which error is worse in HR?**
It depends. An FP ("will accept" but rejects) wastes planning; an FN ("will
reject" but accepts) makes HR miss an opportunity. If late dropout is very
costly, raise recall; if backup planning is expensive, raise precision.

**Q46. What is ROC-AUC?**
The probability that a random positive is ranked above a random negative by the
model's score. 0.5 = random, 1.0 = perfect. It is threshold-independent.

**Q47. Why not report only accuracy?**
With 69.74% negatives, a constant predictor gets 69.74% accuracy while being
useless. Precision, recall, F1 and AUC expose that.

**Q48. What threshold did you use?**
0.5 (`predict` default). It can be tuned; lowering it increases recall.

**Q49. Why choose the model by ROC-AUC?**
AUC evaluates ranking quality over all thresholds and is not affected by the
class imbalance as directly as accuracy; it is a fair comparison across models.

**Q50. Can you show the formulas with your numbers?**
Accuracy = (2549+6265)/10000 = 88.14%. Precision = 2549/3258 = 78.24%.
Recall = 2549/3026 = 84.24%. F1 = 2*0.7824*0.8424/(0.7824+0.8424) = 81.13%.

### F. SQL and deployment

**Q51. Why store the data in a database?**
Persistent, queryable, multi-user storage for reporting; avoids repeatedly
loading and cleaning a large CSV.

**Q52. Which database did you use?**
MySQL through SQLAlchemy + PyMySQL; if the MySQL connection fails, the notebook
automatically falls back to SQLite (`job_acceptance.db`), so it runs anywhere.

**Q53. What tables did you create?**
`candidates` (50,000 rows, cleaned data) and `acceptance_summary_by_tier`
(total, accepted, acceptance rate per company tier).

**Q54. What is SQLAlchemy?**
A Python SQL toolkit/ORM. Here `create_engine` manages the connection and
`to_sql` writes DataFrames; `pd.read_sql` runs queries and returns DataFrames.

**Q55. What does `if_exists="replace"` do?**
Drops and recreates the table on each run, so re-running the notebook does not
duplicate rows.

**Q56. What is Streamlit?**
A Python framework that turns scripts into interactive web apps; widgets
re-run the script on interaction. Here it powers the KPI/EDA/prediction
dashboard.

**Q57. Explain the prediction flow in the app.**
Form input -> DataFrame -> `create_features` -> `get_dummies` -> `reindex` to the
training columns -> `scaler.transform` -> `model.predict_proba` -> probability,
gauge and accept/reject message at 0.5.

**Q58. How would you deploy this for real users?**
Containerise with Docker, host the Streamlit app (Streamlit Community Cloud or a
cloud VM), store `model.pkl` and data securely, add authentication, and monitor
predictions/drift. The current version is a local academic prototype.

### G. Limitations and future work

**Q59. What are the limitations?**
- Single hold-out split, no cross-validation or hyperparameter tuning.
- Dataset is educational and may not represent a specific company.
- Model file is large (132 MB) because of 200 fully grown trees; GitHub excludes
  files over 100 MB.
- No per-candidate explanation (only global feature importance).
- Age and gender are sensitive attributes; fairness/bias must be checked before
  any real HR use.
- No authentication or monitoring in the dashboard; it is local only.
- Imputation statistics (median/mode) were computed on the full cleaned dataset
  before splitting. Strictly, imputation should also be fitted on the training
  set only; scaling is already fitted on train only.

**Q60. What is your future scope?**
5-fold cross-validation and GridSearchCV; compare XGBoost/LightGBM; try SMOTE
and threshold tuning; add SHAP explanations; reduce model size (fewer/shallower
trees or compression); deploy with authentication and monitoring; add an API for
the HR system; validate on real company data.

**Q61. How would you reduce the 132 MB model file?**
Fewer trees, `max_depth` limits, `min_samples_leaf`, or `joblib.dump(...,
compress=3)`. The accuracy/size trade-off would be measured on the test set.

**Q62. Is gender/age fair to use?**
They are in the dataset, but using protected attributes in hiring decisions can
create illegal or unethical bias. Before production I would run a fairness audit
and likely exclude or carefully control them. This is a limitation of the
academic prototype.

**Q63. What did you learn from this project?**
The full data science lifecycle: cleaning and leakage checks, EDA-driven feature
engineering, honest evaluation with imbalance-aware metrics, model persistence,
and turning a model into a usable dashboard. The biggest lesson was that a
"perfect" score usually means leakage, not success.

---

## 9. Numbers to memorise (one-page cheat sheet)

- Raw rows: **51,500** -> cleaned rows: **50,000** (1,500 duplicates removed:
  1,376 exact + 124 after label standardisation).
- Missing values after cleaning: **0**.
- Positive class: **15,130 / 50,000 = 30.26%** Placed.
- Features: **25 raw + 5 engineered = 30**, encoded to **39**.
- Split: **40,000 train / 10,000 test**, stratified, `random_state=42`.
- Models: Logistic Regression, Decision Tree (`max_depth=8`), Random Forest
  (**200 trees**), all `class_weight="balanced"`.
- Random Forest test scores: **Accuracy 88.14%, Precision 78.24%, Recall 84.24%,
  F1 81.13%, ROC-AUC 0.9518**; confusion matrix TP 2,549, TN 6,265, FP 709,
  FN 477.
- Logistic Regression: 85.18% / 70.92% / 86.48% / 77.93% / 0.9371.
- Decision Tree: 81.25% / 63.53% / 89.29% / 74.24% / 0.9135.
- Baseline (always Not Placed): **69.74%** accuracy.
- Top features: technical score 0.246, skills match 0.076, CTC gap 0.075,
  experience 0.052, previous CTC 0.046.
- Key EDA: interview score <50 -> ~0% acceptance; top band (87+) -> ~77%;
  skills Low 11.1% / High 41.4%; Fresher 13.8% / Senior 53.7%; tier is flat
  (~30%).
- Files: `Dataset.csv` -> `Cleaned_Dataset.csv` -> `Final_Dataset.csv` ->
  `model.pkl` -> `app.py`.
- Run app: `streamlit run app.py`.

---

## 10. Five-minute demo script

1. **Start**: `streamlit run app.py` (or show the already running app).
2. **Overview**: say the domain and goal, point at the 7 KPIs, explain
   High-Risk = predicted probability < 50%. Show that acceptance rises with
   interview score and skills match, and that tier is flat.
3. **EDA page**: show the model comparison table, say Random Forest was chosen
   by ROC-AUC (0.9518), then show the confusion matrix/feature importance
   figure and mention technical score dominates.
4. **Predict page**: use the strong profile below -> about **76% acceptance**.
   Then change to the weak profile -> about **0.5%**. Explain that the gauge is
   `predict_proba` and the decision boundary is 0.5.
5. **Close**: one sentence on limitations (single split, educational data,
   fairness, model not deployed) and future work (cross-validation, tuning,
   SHAP, deployment).

Strong demo profile (approx. 76%):

- Age 26, Male, SSC 88, HSC 86, Degree 84, Computer Science
- Technical 92, Aptitude 88, Communication 85, Skills match 93,
  Certifications 5, Internship Yes
- Experience 4 years, Willing to switch, Relevant experience
- Previous CTC 6.0, Expected CTC 8.0, Tier 1, Role Matched, Competition Low,
  No bond, Notice 30, No layoff, Gap 0, Willing to relocate

Weak demo profile (approx. 0.5%):

- Age 31, Female, SSC 60, HSC 58, Degree 60, Others
- Technical 45, Aptitude 42, Communication 50, Skills match 52,
  Certifications 0, No internship
- Fresher, Not willing to switch, Not relevant
- Previous CTC 0, Expected CTC 12, Tier 3, Role Not Matched, Competition High,
  Bond required, Notice 60, Layoff Yes, Gap 12 months, Not willing to relocate

---

## 11. Short code answers (if asked to write on the spot)

**Train a Random Forest and evaluate:**

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

df = pd.read_csv("Final_Dataset.csv")
df["target"] = (df["status"] == "Placed").astype(int)
X = pd.get_dummies(df.drop(columns=["status", "target"]), drop_first=True)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]
print(accuracy_score(y_test, pred), roc_auc_score(y_test, prob))
```

**Predict one candidate from the saved artifacts:**

```python
import joblib
artifacts = joblib.load("model.pkl")
model, scaler = artifacts["model"], artifacts["scaler"]
columns, num_cols = artifacts["columns"], artifacts["num_cols"]

row = pd.get_dummies(new_df, drop_first=True)
row = row.reindex(columns=columns, fill_value=0)
row[num_cols] = scaler.transform(row[num_cols])
print(model.predict_proba(row)[:, 1])
```

**Formulas:**

```python
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)
accuracy  = (TP + TN) / (TP + TN + FP + FN)
```

---

## 12. Last-minute warnings

- Do not say the model is 100% accurate. It was 100% only with leakage; the
  honest result is 88.14%.
- Do not say accuracy is the main metric. Say F1/ROC-AUC because of imbalance.
- Do not claim SMOTE, cross-validation, GridSearch or SHAP were used - they were
  not; present them as future work.
- Do not say correlation proves causation, or that feature importance proves
  causation.
- Do not claim the system makes hiring decisions. It supports HR; a human
  decides.
- If you do not know something, say: "I did not test that in this project; it is
  a limitation and a next step." That is an accepted viva answer.
