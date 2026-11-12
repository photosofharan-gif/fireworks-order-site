from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Student Placement Prediction & Skill Gap Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "lalo-salamanca-jmc-24/student-placement-prediction/"
    "main/Placement_BeginnerTask01.csv"
)
RANDOM_STATE = 42

FEATURES = [
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks",
]

VERIFIED_MODEL_RESULTS = pd.DataFrame(
    [
        ["Logistic Regression", 0.8085, 0.7695, 0.7759, 0.7727, 0.8837],
        ["Random Forest", 0.7915, 0.7638, 0.7282, 0.7456, 0.8712],
        ["K-Nearest Neighbours", 0.7895, 0.7568, 0.7342, 0.7453, 0.8649],
        ["Decision Tree", 0.7865, 0.7682, 0.7032, 0.7343, 0.8632],
    ],
    columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
)

BENCHMARKS = {
    "CGPA": 8.2,
    "Internships": 1.0,
    "Projects": 3.0,
    "Workshops/Certifications": 2.0,
    "AptitudeTestScore": 86.0,
    "SoftSkillsRating": 4.6,
    "ExtracurricularActivities": 1.0,
    "PlacementTraining": 1.0,
}

RECOMMENDATIONS = {
    "CGPA": "Strengthen academic consistency and focus on weak subjects.",
    "Internships": "Gain relevant practical exposure through an internship.",
    "Projects": "Build additional projects that demonstrate applied technical ability.",
    "Workshops/Certifications": "Complete relevant workshops or recognized certifications.",
    "AptitudeTestScore": "Practice quantitative aptitude, logical reasoning and problem-solving.",
    "SoftSkillsRating": "Improve communication, teamwork, presentation and interview skills.",
    "ExtracurricularActivities": "Participate in meaningful extracurricular or leadership activities.",
    "PlacementTraining": "Attend placement training, mock interviews and resume preparation sessions.",
}

LABELS = {
    "CGPA": "CGPA",
    "Internships": "Internships",
    "Projects": "Projects",
    "Workshops/Certifications": "Workshops / Certifications",
    "AptitudeTestScore": "Aptitude Test Score",
    "SoftSkillsRating": "Soft Skills Rating",
    "ExtracurricularActivities": "Extracurricular Activities",
    "PlacementTraining": "Placement Training",
}

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        padding: 0.8rem;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_URL)

@st.cache_resource(show_spinner=False)
def train_selected_model():
    df = load_dataset().copy()
    df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
    df["PlacementTraining"] = df["PlacementTraining"].map({"Yes": 1, "No": 0})
    df["PlacementStatus"] = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

    X = df[FEATURES]
    y = df["PlacementStatus"]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model, df

def build_skill_gap_table(profile):
    rows = []
    for feature, benchmark in BENCHMARKS.items():
        if float(profile[feature]) < float(benchmark):
            rows.append(
                {
                    "Area": LABELS[feature],
                    "Your value": profile[feature],
                    "Reference": benchmark,
                    "Recommendation": RECOMMENDATIONS[feature],
                }
            )
    return pd.DataFrame(rows)

st.title("🎓 Student Placement Prediction and Skill Gap Analysis")
st.caption("BCA VI Semester • Education and Employability Analytics")
st.info(
    "Educational decision-support prototype. A prediction is not a guarantee of employment."
)

with st.sidebar:
    st.header("Project Summary")
    st.write("**Selected model:** Logistic Regression")
    st.write("**Verified accuracy:** 80.85%")
    st.write("**Verified F1-score:** 0.7727")
    st.write("**Dataset:** 10,000 student records")
    st.divider()
    st.caption(
        "The public dataset is not SRM-specific. Results are intended for academic evaluation."
    )

prediction_tab, analytics_tab, methodology_tab, about_tab = st.tabs(
    ["Prediction", "Analytics", "Methodology", "About"]
)

with prediction_tab:
    st.subheader("Student Profile")

    try:
        model, model_df = train_selected_model()
        model_available = True
    except Exception as exc:
        model_available = False
        model = None
        model_df = None
        st.error("The live model could not load the public dataset. Please reload the app.")
        with st.expander("Technical detail"):
            st.code(str(exc))

    with st.form("student_profile_form"):
        left, right = st.columns(2)

        with left:
            cgpa = st.slider("CGPA", 6.5, 9.1, 7.4, 0.1)
            internships = st.slider("Internships completed", 0, 2, 1)
            projects = st.slider("Projects completed", 0, 3, 2)
            workshops = st.slider("Workshops / Certifications", 0, 3, 1)
            aptitude = st.slider("Aptitude Test Score", 60, 90, 78)

        with right:
            soft_skills = st.slider("Soft Skills Rating", 3.0, 4.8, 4.1, 0.1)
            extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"], index=0)
            placement_training = st.selectbox("Placement Training", ["Yes", "No"], index=0)
            ssc_marks = st.slider("SSC Marks", 55, 90, 72)
            hsc_marks = st.slider("HSC Marks", 57, 88, 75)

        submitted = st.form_submit_button(
            "Analyze Placement Readiness",
            use_container_width=True,
            disabled=not model_available,
        )

    if submitted and model_available:
        profile = {
            "CGPA": cgpa,
            "Internships": internships,
            "Projects": projects,
            "Workshops/Certifications": workshops,
            "AptitudeTestScore": aptitude,
            "SoftSkillsRating": soft_skills,
            "ExtracurricularActivities": 1 if extracurricular == "Yes" else 0,
            "PlacementTraining": 1 if placement_training == "Yes" else 0,
            "SSC_Marks": ssc_marks,
            "HSC_Marks": hsc_marks,
        }

        input_df = pd.DataFrame([profile], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])

        st.divider()
        st.subheader("Placement Readiness Result")
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected Model", "Logistic Regression")
        m2.metric("Prediction", "Placement Ready" if prediction == 1 else "Needs Improvement")
        m3.metric("Estimated Probability", f"{probability:.1%}")

        if prediction == 1:
            st.success("The model classifies this profile as placement-ready.")
        else:
            st.warning("The model classifies this profile as needing improvement.")

        st.subheader("Skill Gap Analysis")
        gaps = build_skill_gap_table(profile)
        if gaps.empty:
            st.success("No major gap detected against the dataset-derived reference profile.")
        else:
            st.dataframe(gaps, use_container_width=True, hide_index=True)

        st.caption(
            "References are derived from placed-student medians/modes in the selected dataset; "
            "they are not universal employer cut-offs."
        )

with analytics_tab:
    st.subheader("Verified Model Evaluation")
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", "80.85%")
    b.metric("Precision", "76.95%")
    c.metric("Recall", "77.59%")
    d.metric("F1-score", "0.7727")

    shown = VERIFIED_MODEL_RESULTS.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]:
        shown[col] = shown[col].map(lambda x: f"{x:.4f}")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    st.subheader("Model Comparison")
    st.bar_chart(
        VERIFIED_MODEL_RESULTS.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        ]
    )

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        [[966, 195], [188, 651]],
        index=["Actual NotPlaced", "Actual Placed"],
        columns=["Predicted NotPlaced", "Predicted Placed"],
    )
    st.dataframe(cm_df, use_container_width=True)
    st.write("Correct predictions: **1,617 / 2,000**")

    st.subheader("Skill-Gap Reference Profile")
    st.dataframe(
        pd.DataFrame(
            {
                "Feature": [LABELS[name] for name in BENCHMARKS],
                "Placed-student reference": list(BENCHMARKS.values()),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

with methodology_tab:
    st.subheader("Project Workflow")
    st.markdown(
        """
1. Data collection from a public 10,000-record placement dataset.
2. Data quality checks and exploratory analysis.
3. Categorical encoding and feature preparation.
4. Stratified 80:20 train-test split.
5. Comparison of Logistic Regression, Decision Tree, Random Forest and KNN.
6. Model selection by F1-score.
7. Skill-gap analysis using placed-student median/mode reference values.
8. Transparent improvement recommendations.
        """
    )

    st.subheader("Why Logistic Regression")
    st.write(
        "It achieved the highest F1-score (0.7727) and the highest test accuracy "
        "(80.85%) among the four evaluated classifiers."
    )

with about_tab:
    st.subheader("Project Objective")
    st.write(
        "Estimate student placement readiness using machine learning and provide "
        "transparent, data-derived feedback about selected academic and employability factors."
    )

    st.subheader("Technology")
    st.write("Python • Pandas • Scikit-learn • Streamlit")

    st.subheader("Limitations")
    st.markdown(
        """
- The dataset is public and is not a verified SRM placement dataset.
- Company-specific requirements and interview performance are not included.
- Labour-market conditions and role-specific skills are not represented.
- Skill-gap benchmarks are dataset-specific, not universal hiring standards.
- Predictive association does not prove causation.
        """
    )

    st.subheader("Future Enhancements")
    st.markdown(
        """
- Train with verified institution-specific placement records.
- Add explainable-AI contributions for individual predictions.
- Add resume/NLP analysis and role-specific recommendations.
- Add placement-cell dashboards and prediction history.
        """
    )
