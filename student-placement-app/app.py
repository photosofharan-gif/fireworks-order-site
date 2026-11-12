import pandas as pd
import streamlit as st
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Student Placement Readiness",
    page_icon="🎓",
    layout="centered",
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
    "AptitudeTestScore": "Aptitude Score",
    "SoftSkillsRating": "Soft Skills",
    "ExtracurricularActivities": "Extracurricular Activities",
    "PlacementTraining": "Placement Training",
}

RECOMMENDATIONS = {
    "CGPA": "Focus on improving academic consistency and weak subjects.",
    "Internships": "Gain practical exposure through a relevant internship.",
    "Projects": "Complete one more practical project that demonstrates your technical ability.",
    "Workshops/Certifications": "Complete an additional relevant workshop or certification.",
    "AptitudeTestScore": "Practice quantitative aptitude, logical reasoning and problem-solving regularly.",
    "SoftSkillsRating": "Improve communication, presentation and interview skills through mock practice.",
    "ExtracurricularActivities": "Participate in meaningful extracurricular or leadership activities.",
    "PlacementTraining": "Attend placement training, resume preparation and mock interview sessions.",
}

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 3rem;}
    .app-title {font-size: 2.15rem; font-weight: 800; color: #172033; margin-bottom: 0.2rem;}
    .app-subtitle {color: #667085; margin-bottom: 1.4rem;}
    .result-box {
        border: 1px solid #DCE6F1;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: #F8FBFF;
        margin: 0.8rem 0 1rem;
    }
    .gap-box {
        border: 1px solid #E6EAF0;
        border-left: 4px solid #2563EB;
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        margin: 0.55rem 0;
        background: #FFFFFF;
    }
    .gap-title {font-weight: 700; color: #172033;}
    .gap-meta {font-size: 0.9rem; color: #667085; margin: 0.15rem 0 0.35rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner=False)
def train_model():
    df = load_data().copy()

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
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def to_binary(value):
    return 1 if value == "Yes" else 0


def readiness_level(probability):
    pct = probability * 100
    if pct < 40:
        return "Low"
    if pct < 60:
        return "Moderate"
    if pct < 80:
        return "Good"
    return "High"


def build_gaps(profile):
    gaps = []
    for feature, benchmark in BENCHMARKS.items():
        value = float(profile[feature])
        if value < float(benchmark):
            gaps.append(
                {
                    "feature": LABELS[feature],
                    "current": profile[feature],
                    "reference": benchmark,
                    "recommendation": RECOMMENDATIONS[feature],
                }
            )
    return gaps


try:
    model = train_model()
    model_ready = True
except Exception as exc:
    model = None
    model_ready = False
    startup_error = str(exc)


st.markdown('<div class="app-title">🎓 Student Placement Readiness</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Check your placement readiness, identify skill gaps and get a simple improvement plan.</div>',
    unsafe_allow_html=True,
)

if not model_ready:
    st.error("The prediction service could not be started. Please refresh the page.")
    with st.expander("Technical details"):
        st.code(startup_error)
    st.stop()

with st.form("placement_form"):
    st.subheader("Student Details")
    c1, c2 = st.columns(2)
    with c1:
        student_name = st.text_input("Student Name *", placeholder="Enter your name")
    with c2:
        register_number = st.text_input("Register Number *", placeholder="Enter register number")

    course = st.text_input("Course", value="BCA", disabled=True)

    st.divider()
    st.subheader("Academic and Placement Profile")
    st.caption("Enter your current details within the supported ranges used by the prediction model.")

    left, right = st.columns(2)

    with left:
        cgpa = st.slider("CGPA", 6.5, 9.1, 7.4, 0.1)
        ssc_marks = st.slider("SSC Marks", 55, 90, 72)
        hsc_marks = st.slider("HSC Marks", 57, 88, 75)
        internships = st.slider("Internships Completed", 0, 2, 1)
        projects = st.slider("Projects Completed", 0, 3, 2)

    with right:
        workshops = st.slider("Workshops / Certifications", 0, 3, 1)
        aptitude = st.slider("Aptitude Test Score", 60, 90, 78)
        soft_skills = st.slider("Soft Skills Rating", 3.0, 4.8, 4.1, 0.1)
        extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
        placement_training = st.selectbox("Placement Training", ["Yes", "No"])

    submitted = st.form_submit_button(
        "Analyze My Placement Readiness",
        use_container_width=True,
    )

if submitted:
    if not student_name.strip() or not register_number.strip():
        st.error("Please enter both Student Name and Register Number.")
        st.stop()

    profile = {
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Workshops/Certifications": workshops,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": soft_skills,
        "ExtracurricularActivities": to_binary(extracurricular),
        "PlacementTraining": to_binary(placement_training),
        "SSC_Marks": ssc_marks,
        "HSC_Marks": hsc_marks,
    }

    input_df = pd.DataFrame([profile], columns=FEATURES)
    probability = float(model.predict_proba(input_df)[0, 1])
    prediction = int(model.predict(input_df)[0])
    probability_pct = probability * 100
    level = readiness_level(probability)
    gaps = build_gaps(profile)

    st.divider()
    st.subheader("Placement Readiness Report")

    st.markdown(
        f"""
        <div class="result-box">
            <b>Student</b> &nbsp; {student_name}<br>
            <b>Register Number</b> &nbsp; {register_number}<br>
            <b>Course</b> &nbsp; BCA
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        "Result",
        "Placement Ready" if prediction == 1 else "Needs Improvement",
    )
    m2.metric("Readiness Score", f"{probability_pct:.1f}%")
    m3.metric("Readiness Level", level)

    st.progress(min(max(probability, 0.0), 1.0))

    if prediction == 1:
        st.success("Your profile shows good placement readiness based on the selected model.")
    else:
        st.warning("Your profile can be improved. Focus on the skill gaps listed below.")

    st.subheader("Skill Gap Analysis")

    if not gaps:
        st.success("No major gap was detected against the dataset-derived reference profile.")
    else:
        for gap in gaps:
            st.markdown(
                f"""
                <div class="gap-box">
                    <div class="gap-title">{gap["feature"]}</div>
                    <div class="gap-meta">
                        Your value: <b>{gap["current"]}</b> &nbsp; | &nbsp;
                        Reference: <b>{gap["reference"]}</b>
                    </div>
                    {gap["recommendation"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Your Action Plan")
    if gaps:
        for index, gap in enumerate(gaps, start=1):
            st.write(f"{index}. {gap['recommendation']}")
    else:
        st.write("Maintain your current preparation and continue practicing for aptitude, technical and interview rounds.")

    report_lines = [
        "STUDENT PLACEMENT READINESS REPORT",
        "",
        f"Student Name: {student_name}",
        f"Register Number: {register_number}",
        "Course: BCA",
        "",
        f"Placement Result: {'Placement Ready' if prediction == 1 else 'Needs Improvement'}",
        f"Readiness Score: {probability_pct:.1f}%",
        f"Readiness Level: {level}",
        "",
        "SKILL GAP ANALYSIS",
    ]

    if gaps:
        for gap in gaps:
            report_lines.extend(
                [
                    f"- {gap['feature']}",
                    f"  Current: {gap['current']} | Reference: {gap['reference']}",
                    f"  Recommendation: {gap['recommendation']}",
                ]
            )
    else:
        report_lines.append("- No major skill gap detected against the dataset-derived reference profile.")

    report_lines.extend(
        [
            "",
            "Note: This is an educational decision-support result based on the selected public dataset.",
            "It is not a guarantee of employment.",
        ]
    )

    report_text = "\n".join(report_lines)

    st.download_button(
        "Download My Result",
        data=report_text,
        file_name=f"{register_number.strip()}_placement_readiness.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.caption(
        "The prediction is an educational decision-support estimate. "
        "Skill-gap references are derived from the selected dataset and are not universal employer cut-offs."
    )
