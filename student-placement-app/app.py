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
SRM_LOGO_B64 = "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[...truncated...]
"
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
        <img src="data:image/png;base64,{SRM_LOGO_B64}" style="width:220px;max-width:45%;margin-bottom:0.9rem" />
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
