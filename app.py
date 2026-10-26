import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Job Acceptance Prediction System", page_icon="📊", layout="wide")

# ---------------- loading data and model ----------------
df = pd.read_csv("Cleaned_Dataset.csv")
df["interview_avg"] = df[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1)

artifacts = joblib.load("model.pkl")
model = artifacts["model"]
scaler = artifacts["scaler"]
model_columns = artifacts["columns"]
num_cols = artifacts["num_cols"]


def create_features(data):
    """same feature engineering as notebook 03"""
    data["experience_category"] = pd.cut(data["years_of_experience"],
                                         bins=[-1, 0, 2, 10],
                                         labels=["Fresher", "Junior", "Senior"]).astype(str)
    data["academic_performance_band"] = pd.cut(
        data[["ssc_percentage", "hsc_percentage", "degree_percentage"]].mean(axis=1),
        bins=[0, 60, 75, 100], labels=["Low", "Medium", "High"]).astype(str)
    data["skills_match_level"] = pd.cut(data["skills_match_percentage"],
                                        bins=[0, 60, 85, 100],
                                        labels=["Low", "Medium", "High"]).astype(str)
    data["interview_performance_category"] = pd.cut(
        data[["technical_score", "aptitude_score", "communication_score"]].mean(axis=1),
        bins=[0, 55, 70, 100], labels=["Weak", "Average", "Strong"]).astype(str)
    data["ctc_expectation_gap_lpa"] = data["expected_ctc_lpa"] - data["previous_ctc_lpa"]
    return data


def predict_acceptance(raw_df):
    """preprocess exactly like in notebook 05, then predict"""
    data = create_features(raw_df.copy())
    data = pd.get_dummies(data, drop_first=True)
    data = data.reindex(columns=model_columns, fill_value=0)   # same columns as training
    data[num_cols] = scaler.transform(data[num_cols])
    return model.predict_proba(data)[:, 1]


# acceptance probability of every candidate (used in the High-Risk KPI)
all_probs = predict_acceptance(df.drop(columns=["status"]))

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard Overview", "🔍 EDA Insights", "🎯 Predict Candidate"])
st.sidebar.write("Model: Random Forest | Records:", len(df))

# ============================================================ OVERVIEW
if page == "📊 Dashboard Overview":
    st.title("📊 Job Acceptance Prediction System")
    st.write("**Domain:** HR Analytics & Predictive Modeling | "
             "**Goal:** predict whether a candidate will accept a job offer")
    st.divider()

    placed_rate = (df["status"] == "Placed").mean() * 100
    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
    kpi1.metric("Total Candidates", f"{len(df):,}")
    kpi2.metric("Placement Rate (%)", f"{placed_rate:.1f}%")
    kpi3.metric("Job Acceptance Rate (%)", f"{placed_rate:.1f}%")
    kpi4.metric("Avg Interview Score", f"{df['interview_avg'].mean():.1f}")
    kpi5.metric("Avg Skills Match (%)", f"{df['skills_match_percentage'].mean():.1f}%")
    kpi6.metric("Offer Dropout Rate (%)", f"{100 - placed_rate:.1f}%")
    kpi7.metric("High-Risk Candidates (%)", f"{(all_probs < 0.5).mean() * 100:.1f}%")
    st.caption("High-Risk = candidates whose predicted acceptance probability is below 50%. "
               "Offer Dropout = candidates who did not accept the offer (Not Placed).")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Acceptance rate by company tier")
        tier_rate = (df.assign(p=df["status"] == "Placed")
                     .groupby("company_tier")["p"].mean().mul(100).reset_index(name="rate"))
        fig = px.bar(tier_rate, x="company_tier", y="rate", color="company_tier",
                     labels={"rate": "Acceptance rate (%)", "company_tier": "Company tier"},
                     text=tier_rate["rate"].round(1))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Candidate status split")
        fig = px.pie(df["status"].value_counts().reset_index(), names="status", values="count",
                     color="status", hole=0.45,
                     color_discrete_map={"Placed": "#2a9d8f", "Not Placed": "#e76f51"})
        st.plotly_chart(fig, width="stretch")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Placement probability vs interview score")
        tmp = df.copy()
        tmp["interview_band"] = pd.cut(tmp["interview_avg"], bins=8)
        band = (tmp.assign(p=tmp["status"] == "Placed")
                  .groupby("interview_band", observed=True)["p"].mean().mul(100).reset_index(name="rate"))
        band["mid"] = [(i.left + i.right) / 2 for i in band["interview_band"]]
        fig = px.line(band, x="mid", y="rate", markers=True,
                      labels={"mid": "Interview score", "rate": "Acceptance rate (%)"})
        st.plotly_chart(fig, width="stretch")
    with col4:
        st.subheader("Acceptance rate by skills match level")
        tmp = df.copy()
        tmp["skills_level"] = pd.cut(tmp["skills_match_percentage"], bins=[0, 60, 85, 100],
                                     labels=["Low (<60)", "Medium (60-85)", "High (>85)"])
        sk = (tmp.assign(p=tmp["status"] == "Placed")
                .groupby("skills_level", observed=True)["p"].mean().mul(100).reset_index(name="rate"))
        fig = px.bar(sk, x="skills_level", y="rate", color="skills_level",
                     color_discrete_sequence=["#e76f51", "#e9c46a", "#2a9d8f"],
                     labels={"rate": "Acceptance rate (%)", "skills_level": "Skills match level"})
        st.plotly_chart(fig, width="stretch")

# ============================================================ EDA
elif page == "🔍 EDA Insights":
    st.title("🔍 Exploratory Data Analysis")
    st.caption("Charts from notebook 02 - Exploratory_Data_Analysis.ipynb and "
               "model results from notebook 05 - Model_Building.ipynb")

    results = pd.DataFrame([
        {"Model": "Random Forest", "Accuracy": 0.8814, "Precision": 0.7881,
         "Recall": 0.8410, "F1": 0.8113, "ROC_AUC": 0.9518},
        {"Model": "Logistic Regression", "Accuracy": 0.8518, "Precision": 0.7093,
         "Recall": 0.8653, "F1": 0.7793, "ROC_AUC": 0.9371},
        {"Model": "Decision Tree", "Accuracy": 0.8125, "Precision": 0.6364,
         "Recall": 0.8928, "F1": 0.7424, "ROC_AUC": 0.9135},
    ]).set_index("Model")
    st.subheader("Model comparison (test set)")
    st.dataframe(results.style.format("{:.2%}").format({"ROC_AUC": "{:.4f}"}), width="stretch")
    st.write("Random Forest is selected because it has the highest ROC-AUC (0.9518).")
    st.divider()

    fig_files = ["1_target_distribution.png", "2_acceptance_by_category.png",
                 "3_interview_vs_acceptance.png", "4_skills_match_impact.png",
                 "5_experience_vs_placement.png", "6_correlation_heatmap.png",
                 "7_feature_importance.png"]
    for i in range(0, len(fig_files), 2):
        cols = st.columns(2)
        for col, fname in zip(cols, fig_files[i:i + 2]):
            col.image(f"figures/{fname}", caption=fname.replace(".png", "").replace("_", " "),
                      width="stretch")

# ============================================================ PREDICT
elif page == "🎯 Predict Candidate":
    st.title("🎯 Candidate Job-Acceptance Prediction")
    st.write("Enter the candidate details and the model will predict if they will accept the offer.")

    with st.form("predict_form"):
        st.subheader("Candidate details")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age", 21, 34, 27)
            gender = st.selectbox("Gender", ["Male", "Female"])
            ssc = st.slider("SSC percentage", 50.0, 95.0, 70.0)
            hsc = st.slider("HSC percentage", 50.0, 95.0, 72.0)
            degree = st.slider("Degree percentage", 55.0, 95.0, 74.0)
            specialization = st.selectbox("Degree specialization",
                                          ["Computer Science", "Electronics",
                                           "Information Technology", "Mechanical", "Others"])
        with c2:
            technical = st.slider("Technical score", 40.0, 100.0, 68.0)
            aptitude = st.slider("Aptitude score", 40.0, 100.0, 64.0)
            communication = st.slider("Communication score", 45.0, 100.0, 66.0)
            skills_match = st.slider("Skills match %", 45.0, 100.0, 74.0)
            certifications = st.slider("Certifications count", 0, 8, 1)
            internship = st.selectbox("Internship experience", ["Yes", "No"])
        with c3:
            experience = st.slider("Years of experience", 0, 5, 1)
            career_switch = st.selectbox("Career switch willingness", ["Willing", "Not Willing"])
            relevant_exp = st.selectbox("Relevant experience", ["Relevant", "Not Relevant"])
            previous_ctc = st.slider("Previous CTC (LPA)", 0.0, 13.8, 4.8)
            expected_ctc = st.slider("Expected CTC (LPA)", 3.0, 26.5, 8.0)
            company_tier = st.selectbox("Company tier", ["Tier 1", "Tier 2", "Tier 3"])
            role_match = st.selectbox("Job role match", ["Matched", "Not Matched"])
            competition = st.selectbox("Competition level", ["Low", "Medium", "High"])
            bond = st.selectbox("Bond requirement", ["Not Required", "Required"])
            notice = st.slider("Notice period (days)", 0, 90, 20)
            layoff = st.selectbox("Layoff history", ["No", "Yes"])
            gap = st.slider("Employment gap (months)", 0, 18, 0)
            relocation = st.selectbox("Relocation willingness", ["Willing", "Not Willing"])
        submit = st.form_submit_button("Predict Acceptance", type="primary")

    if submit:
        input_df = pd.DataFrame([{
            "age_years": age, "gender": gender, "ssc_percentage": ssc,
            "hsc_percentage": hsc, "degree_percentage": degree,
            "degree_specialization": specialization, "technical_score": technical,
            "aptitude_score": aptitude, "communication_score": communication,
            "skills_match_percentage": skills_match, "certifications_count": certifications,
            "internship_experience": internship, "years_of_experience": experience,
            "career_switch_willingness": career_switch, "relevant_experience": relevant_exp,
            "previous_ctc_lpa": previous_ctc, "expected_ctc_lpa": expected_ctc,
            "company_tier": company_tier, "job_role_match": role_match,
            "competition_level": competition, "bond_requirement": bond,
            "notice_period_days": notice, "layoff_history": layoff,
            "employment_gap_months": gap, "relocation_willingness": relocation,
        }])

        prob = predict_acceptance(input_df)[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            if prob >= 0.5:
                st.success("## ✅ Likely to ACCEPT the offer")
            else:
                st.error("## ❌ Likely to REJECT the offer")
            st.metric("Acceptance probability", f"{prob * 100:.1f}%")
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=prob * 100, number={"suffix": "%"},
                title={"text": "Offer Acceptance Probability"},
                gauge={"axis": {"range": [0, 100]},
                       "bar": {"color": "#2a9d8f" if prob >= 0.5 else "#e76f51"},
                       "steps": [{"range": [0, 50], "color": "#fdecea"},
                                 {"range": [50, 100], "color": "#e8f6f3"}],
                       "threshold": {"line": {"color": "black", "width": 3},
                                     "thickness": 0.75, "value": 50}}))
            st.plotly_chart(fig, width="stretch")

