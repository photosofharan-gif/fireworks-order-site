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

DATA_PATH = Path(__file__).parent / "Placement_BeginnerTask01.csv"
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

CORRELATIONS = pd.DataFrame(
    [
        ["Aptitude Test Score", 0.522],
        ["HSC Marks", 0.506],
        ["Projects", 0.475],
        ["SSC Marks", 0.470],
        ["Soft Skills Rating", 0.427],
        ["CGPA", 0.423],
        ["Workshops / Certifications", 0.362],
        ["Internships", 0.260],
    ],
    columns=["Feature", "Correlation"],
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

RECOMMENDATIONS = {
    "CGPA": "Strengthen academic consistency and focus on weak subjects.",
    "Internships": "Gain relevant practical exposure through an internship.",
    "Projects": "Build one more practical project that demonstrates applied technical ability.",
    "Workshops/Certifications": "Complete a relevant workshop or recognized certification.",
    "AptitudeTestScore": "Practice quantitative aptitude, logical reasoning and problem-solving.",
    "SoftSkillsRating": "Improve communication, teamwork, presentation and interview skills.",
    "ExtracurricularActivities": "Participate in meaningful extracurricular or leadership activities.",
    "PlacementTraining": "Attend placement training, mock interviews and resume preparation sessions.",
}

st.markdown(
    """
    <style>
    :root {
        --brand-blue: #034DA1;
        --brand-blue-dark: #063970;
        --ink: #172033;
        --muted: #667085;
        --line: #E6EAF0;
        --soft: #F5F8FC;
        --success: #0F8A5F;
        --warning: #B86B00;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background: linear-gradient(180deg, #F8FAFD 0%, #FFFFFF 34%);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--line);
        background: #FFFFFF;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.8rem;
    }

    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(20, 40, 80, 0.04);
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted);
        font-weight: 600;
    }

    div[data-testid="stMetricValue"] {
        color: var(--ink);
    }

    div[data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero {
        background: linear-gradient(135deg, #FFFFFF 0%, #EEF5FD 100%);
        border: 1px solid #DCE9F8;
        border-radius: 20px;
        padding: 1.45rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 28px rgba(3, 77, 161, 0.08);
    }

    .hero-kicker {
        display: inline-block;
        color: #034DA1;
        background: #E8F1FB;
        border: 1px solid #C9DDF5;
        border-radius: 999px;
        padding: 0.28rem 0.7rem;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: 0.75rem;
    }

    .hero-title {
        color: #172033;
        font-size: clamp(1.8rem, 3vw, 2.75rem);
        font-weight: 800;
        line-height: 1.12;
        margin: 0;
    }

    .hero-subtitle {
        color: #667085;
        font-size: 1rem;
        margin-top: 0.6rem;
        margin-bottom: 0;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 5px 18px rgba(20, 40, 80, 0.035);
    }

    .result-card {
        background: linear-gradient(135deg, #F7FAFF 0%, #FFFFFF 100%);
        border: 1px solid #D9E7F7;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin: 0.8rem 0 1rem;
    }

    .readiness-track {
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: #E9EEF5;
        overflow: hidden;
        margin-top: 0.55rem;
    }

    .readiness-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #034DA1, #4E8DD6);
    }

    .gap-card {
        background: #FFFFFF;
        border: 1px solid #E3E8EF;
        border-left: 4px solid #034DA1;
        border-radius: 12px;
        padding: 0.8rem 0.95rem;
        margin: 0.55rem 0;
    }

    .gap-title {
        color: #172033;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .gap-meta {
        color: #667085;
        font-size: 0.9rem;
        margin-bottom: 0.35rem;
    }

    .gap-text {
        color: #344054;
        font-size: 0.95rem;
    }

    .mini-note {
        color: #667085;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    .sidebar-brand {
        color: #034DA1;
        font-weight: 800;
        font-size: 1.08rem;
        margin-bottom: 0.25rem;
    }

    .sidebar-rule {
        border-top: 1px solid #E6EAF0;
        margin: 1rem 0;
    }

    .badge {
        display: inline-block;
        padding: 0.22rem 0.58rem;
        border-radius: 999px;
        background: #E8F1FB;
        color: #034DA1;
        font-size: 0.78rem;
        font-weight: 700;
    }

    .stButton > button, .stFormSubmitButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 2.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_selected_model():
    df = load_dataset().copy()

    df["ExtracurricularActivities"] = df["ExtracurricularActivities"].map({"Yes": 1, "No": 0})
    df["PlacementTraining"] = df["PlacementTraining"].map({"Yes": 1, "No": 0})
    df["PlacementStatus"] = df["PlacementStatus"].map({"Placed": 1, "NotPlaced": 0})

    X = df[FEATURES]
    y = df["PlacementStatus"]

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model, df


def build_skill_gaps(profile):
    rows = []
    for feature, benchmark in BENCHMARKS.items():
        value = float(profile[feature])
        if value < float(benchmark):
            rows.append(
                {
                    "Area": LABELS[feature],
                    "Current": profile[feature],
                    "Reference": benchmark,
                    "Recommendation": RECOMMENDATIONS[feature],
                }
            )
    return rows


def yes_no(value):
    return 1 if value == "Yes" else 0


# Header
st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">SRM • BCA VI SEMESTER PROJECT</div>
        <div class="hero-title">🎓 Student Placement Prediction & Skill Gap Analysis</div>
        <div class="hero-subtitle">
            Machine-learning decision support for placement readiness, skill-gap identification,
            and targeted improvement recommendations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    "Educational decision-support prototype. Predictions are based on the selected public dataset "
    "and should not be treated as a guarantee of employment."
)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-brand">Project Dashboard</div>', unsafe_allow_html=True)
    st.caption("Education & Employability Analytics")

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)

    st.metric("Selected Model", "Logistic Regression")
    st.metric("Verified Accuracy", "80.85%")
    st.metric("Verified F1-score", "0.7727")
    st.metric("Dataset Records", "10,000")

    st.markdown('<div class="sidebar-rule"></div>', unsafe_allow_html=True)
    st.markdown('<span class="badge">MODEL VERIFIED IN COLAB</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="mini-note" style="margin-top:0.8rem">
        The public dataset is used for academic demonstration and is not a verified SRM placement dataset.
        </p>
        """,
        unsafe_allow_html=True,
    )

prediction_tab, analytics_tab, methodology_tab, about_tab = st.tabs(
    ["🎯 Prediction", "📊 Analytics", "🧭 Methodology", "ℹ️ About"]
)

# Load model once
try:
    model, model_df = train_selected_model()
    model_available = True
except Exception as exc:
    model_available = False
    model = None
    model_df = None

# Prediction tab
with prediction_tab:
    st.subheader("Student Profile")
    st.caption("Enter the student's current academic and employability-related attributes.")

    if not model_available:
        st.error("The model could not be initialized. Please reload the application.")
        with st.expander("Technical detail"):
            st.code(str(exc))

    with st.container(border=True):
        with st.form("student_profile_form"):
            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("**Academic & Experience**")
                cgpa = st.slider("CGPA", 6.5, 9.1, 7.4, 0.1)
                internships = st.slider("Internships completed", 0, 2, 1)
                projects = st.slider("Projects completed", 0, 3, 2)
                workshops = st.slider("Workshops / Certifications", 0, 3, 1)
                aptitude = st.slider("Aptitude Test Score", 60, 90, 78)

            with right:
                st.markdown("**Skills & Training**")
                soft_skills = st.slider("Soft Skills Rating", 3.0, 4.8, 4.1, 0.1)
                extracurricular = st.selectbox(
                    "Extracurricular Activities", ["Yes", "No"], index=0
                )
                placement_training = st.selectbox(
                    "Placement Training", ["Yes", "No"], index=0
                )
                ssc_marks = st.slider("SSC Marks", 55, 90, 72)
                hsc_marks = st.slider("HSC Marks", 57, 88, 75)

            submitted = st.form_submit_button(
                "Analyze Placement Readiness",
                width="stretch",
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
            "ExtracurricularActivities": yes_no(extracurricular),
            "PlacementTraining": yes_no(placement_training),
            "SSC_Marks": ssc_marks,
            "HSC_Marks": hsc_marks,
        }

        input_df = pd.DataFrame([profile], columns=FEATURES)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])
        probability_pct = probability * 100
        gaps = build_skill_gaps(profile)

        st.markdown("---")
        st.subheader("Placement Readiness Result")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Model", "Logistic Regression")
        m2.metric(
            "Prediction",
            "Placement Ready" if prediction == 1 else "Needs Improvement",
        )
        m3.metric("Estimated Probability", f"{probability_pct:.1f}%")
        m4.metric("Skill Gaps Identified", len(gaps))

        result_label = (
            "Higher placement readiness" if prediction == 1 else "Improvement recommended"
        )
        st.markdown(
            f"""
            <div class="result-card">
                <strong style="color:#172033;font-size:1.05rem">{result_label}</strong>
                <div class="mini-note" style="margin-top:0.3rem">
                    Estimated placement probability from the selected Logistic Regression model
                </div>
                <div class="readiness-track">
                    <div class="readiness-fill" style="width:{probability_pct:.1f}%"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.35rem;
                            color:#667085;font-size:0.84rem">
                    <span>0%</span><span><strong>{probability_pct:.1f}%</strong></span><span>100%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if prediction == 1:
            st.success(
                "The model classifies this profile as placement-ready based on patterns in the selected dataset."
            )
        else:
            st.warning(
                "The model classifies this profile as needing improvement. The skill-gap section below shows the main actionable areas."
            )

        st.subheader("Skill Gap Analysis")

        if not gaps:
            st.success(
                "No major actionable gap was detected against the dataset-derived placed-student reference profile."
            )
        else:
            for gap in gaps:
                st.markdown(
                    f"""
                    <div class="gap-card">
                        <div class="gap-title">{gap["Area"]}</div>
                        <div class="gap-meta">
                            Current value: <strong>{gap["Current"]}</strong>
                            &nbsp; • &nbsp;
                            Dataset reference: <strong>{gap["Reference"]}</strong>
                        </div>
                        <div class="gap-text">{gap["Recommendation"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.caption(
            "Reference values are medians/modes from students labelled Placed in the selected dataset. "
            "They are not universal employer cut-offs."
        )


# Analytics tab
with analytics_tab:
    st.subheader("Verified Model Performance")
    st.caption("These results come from the executed Google Colab experiment using the held-out test set.")

    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", "80.85%")
    b.metric("Precision", "76.95%")
    c.metric("Recall", "77.59%")
    d.metric("F1-score", "0.7727")
    e.metric("ROC AUC", "0.8837")

    st.markdown("#### Model Comparison")
    shown = VERIFIED_MODEL_RESULTS.copy()
    shown["Accuracy"] = shown["Accuracy"].map(lambda x: f"{x:.2%}")
    shown["Precision"] = shown["Precision"].map(lambda x: f"{x:.2%}")
    shown["Recall"] = shown["Recall"].map(lambda x: f"{x:.2%}")
    shown["F1 Score"] = shown["F1 Score"].map(lambda x: f"{x:.4f}")
    shown["ROC AUC"] = shown["ROC AUC"].map(lambda x: f"{x:.4f}")
    st.dataframe(shown, width="stretch", hide_index=True)

    chart_df = VERIFIED_MODEL_RESULTS.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]
    st.bar_chart(chart_df)

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Confusion Matrix")
        cm_df = pd.DataFrame(
            [[966, 195], [188, 651]],
            index=["Actual NotPlaced", "Actual Placed"],
            columns=["Predicted NotPlaced", "Predicted Placed"],
        )
        st.dataframe(cm_df, width="stretch")
        st.caption("Correct predictions: 1,617 out of 2,000 test records.")

    with right:
        st.markdown("#### Strongest Observed Associations")
        corr_chart = CORRELATIONS.set_index("Feature")
        st.bar_chart(corr_chart)

    st.markdown("#### Dataset-Derived Skill Reference Profile")
    reference_df = pd.DataFrame(
        {
            "Feature": [LABELS[name] for name in BENCHMARKS],
            "Placed-student reference": list(BENCHMARKS.values()),
        }
    )
    st.dataframe(reference_df, width="stretch", hide_index=True)


# Methodology tab
with methodology_tab:
    st.subheader("Project Methodology")

    stages = [
        ("01", "Data Collection", "Public campus placement dataset containing 10,000 student records."),
        ("02", "Data Quality", "Missing-value, duplicate, schema and target-distribution checks."),
        ("03", "EDA", "Descriptive statistics, group comparisons and placement-related associations."),
        ("04", "Preprocessing", "Binary encoding, identifier removal and feature preparation."),
        ("05", "Model Training", "Stratified 80:20 split and four supervised classification algorithms."),
        ("06", "Model Evaluation", "Accuracy, precision, recall, F1-score, ROC AUC and confusion matrix."),
        ("07", "Skill-Gap Analysis", "Placed-student medians/modes used as transparent reference values."),
        ("08", "Application", "Streamlit interface for prediction, recommendations and analytics."),
    ]

    for number, title, text in stages:
        st.markdown(
            f"""
            <div class="gap-card">
                <div class="gap-title">{number} &nbsp; {title}</div>
                <div class="gap-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Why Logistic Regression?")
    st.write(
        "It was selected from the executed experiment because it achieved the highest F1-score "
        "(0.7727) and the highest test accuracy (80.85%) among Logistic Regression, Decision Tree, "
        "Random Forest and K-Nearest Neighbours."
    )

    st.markdown("#### Responsible Interpretation")
    st.write(
        "The model is intended for academic demonstration and student decision support. "
        "It should not be used as a high-stakes employment screening system."
    )


# About tab
with about_tab:
    st.subheader("Project Objective")
    st.write(
        "To estimate student placement readiness using machine learning and provide transparent, "
        "data-derived feedback about selected academic and employability factors."
    )

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Technology Stack")
        st.markdown(
            """
            - Python
            - Pandas
            - Scikit-learn
            - Streamlit
            - Google Colab
            """
        )

        st.markdown("#### Selected Dataset")
        st.write(
            "10,000 student records with academic, experiential, aptitude, soft-skill, "
            "training and placement-status attributes."
        )

    with right:
        st.markdown("#### Limitations")
        st.markdown(
            """
            - Public educational dataset; not verified SRM placement records
            - No company-specific interview criteria
            - No labour-market or role-specific variables
            - Skill-gap references are dataset-specific
            - Predictive association does not establish causation
            """
        )

        st.markdown("#### Future Enhancements")
        st.markdown(
            """
            - Institution-specific verified placement data
            - Explainable-AI contributions for individual predictions
            - Resume/NLP analysis
            - Role-specific recommendations
            - Placement-cell administrator dashboard
            """
        )

st.markdown("---")
st.caption(
    "BCA VI Semester Project • Machine Learning-Based Student Placement Prediction and Skill Gap Analysis System"
)
