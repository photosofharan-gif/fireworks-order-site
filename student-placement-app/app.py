import io
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

APP_NAME = "CareerReady"
APP_SUBTITLE = "Student Placement Readiness & Skill Gap Analysis"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎓",
    layout="wide",
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
    "CGPA": "Improve academic consistency and focus on weak subjects.",
    "Internships": "Gain practical exposure through a relevant internship.",
    "Projects": "Complete one more practical project that demonstrates applied technical ability.",
    "Workshops/Certifications": "Complete an additional relevant workshop or certification.",
    "AptitudeTestScore": "Practice quantitative aptitude, logical reasoning and problem-solving regularly.",
    "SoftSkillsRating": "Improve communication, presentation and interview skills through mock practice.",
    "ExtracurricularActivities": "Participate in meaningful extracurricular or leadership activities.",
    "PlacementTraining": "Attend placement training, resume preparation and mock interview sessions.",
}

COURSES = [
    "BCA - Bachelor of Computer Applications",
    "B.Sc Computer Science",
    "B.Sc Data Science",
    "B.Sc Information Technology",
    "B.Tech Computer Science & Engineering",
    "B.Tech Information Technology",
    "B.Tech Artificial Intelligence & Data Science",
    "B.Com",
    "BBA",
    "BA",
    "MCA - Master of Computer Applications",
    "M.Sc Computer Science",
    "M.Sc Data Science",
    "MBA",
    "M.Tech Computer Science & Engineering",
    "Diploma in Computer Applications",
    "Other",
]

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background: #F7F9FC;}
    .block-container {
        max-width: 1120px;
        padding-top: 1.1rem;
        padding-bottom: 6rem;
    }
    .cr-header {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.3rem 0 1.25rem 0;
        border-bottom: 1px solid #E5EAF0;
        margin-bottom: 1.5rem;
    }
    .cr-logo {
        width: 46px;
        height: 46px;
        border-radius: 12px;
        background: #1457D9;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.45rem;
        box-shadow: 0 4px 12px rgba(20,87,217,.18);
    }
    .cr-name {
        font-size: 1.35rem;
        line-height: 1.15;
        font-weight: 800;
        color: #172033;
    }
    .cr-subtitle {
        color: #667085;
        font-size: 0.88rem;
        margin-top: 0.18rem;
    }
    .page-title {
        font-size: 1.9rem;
        line-height: 1.2;
        font-weight: 800;
        color: #172033;
        margin-bottom: 0.35rem;
    }
    .page-copy {
        color: #667085;
        max-width: 760px;
        margin-bottom: 1.35rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #FFFFFF;
        border-color: #E5EAF0 !important;
        border-radius: 14px;
        box-shadow: 0 4px 14px rgba(16,24,40,.035);
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 12px;
        padding: 0.8rem 0.9rem;
    }
    .summary-card {
        background: linear-gradient(135deg, #F3F7FF 0%, #FFFFFF 100%);
        border: 1px solid #DCE7FB;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        margin: 0.7rem 0 1rem;
    }
    .status-label {
        color: #667085;
        font-size: 0.82rem;
        font-weight: 650;
        text-transform: uppercase;
        letter-spacing: .04em;
    }
    .status-value {
        color: #172033;
        font-size: 1.35rem;
        font-weight: 800;
        margin-top: .2rem;
    }
    .gap-card {
        height: 100%;
        background: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 14px;
        padding: 1rem;
    }
    .gap-area {
        color: #172033;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: .45rem;
    }
    .gap-number {
        color: #1457D9;
        font-size: 1.1rem;
        font-weight: 800;
    }
    .gap-reference {
        color: #667085;
        font-size: .84rem;
        margin-bottom: .55rem;
    }
    .gap-reco {
        color: #344054;
        font-size: .91rem;
        line-height: 1.42;
    }
    .priority {
        display: inline-block;
        padding: .18rem .5rem;
        border-radius: 999px;
        font-size: .74rem;
        font-weight: 750;
        margin-bottom: .45rem;
    }
    .priority-high {background:#FEECEC;color:#B42318;}
    .priority-medium {background:#FFF4E5;color:#B54708;}
    .priority-low {background:#EEF4FF;color:#175CD3;}
    .strength-chip {
        display: inline-block;
        padding: .35rem .62rem;
        margin: .2rem .25rem .2rem 0;
        border-radius: 999px;
        background: #ECFDF3;
        color: #027A48;
        border: 1px solid #ABEFC6;
        font-size: .82rem;
        font-weight: 650;
    }
    .action-card {
        background: #FFFFFF;
        border: 1px solid #E5EAF0;
        border-radius: 12px;
        padding: .9rem 1rem;
        margin-bottom: .65rem;
    }
    .action-title {font-weight: 750; color:#172033;}
    .action-copy {color:#475467;font-size:.91rem;margin-top:.2rem;}
    .note {
        color: #667085;
        font-size: .82rem;
        line-height: 1.45;
    }
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
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    return model


def yes_no(value):
    return 1 if value == "Yes" else 0


def gap_priority(current, reference):
    ratio = 0 if reference == 0 else float(current) / float(reference)
    if ratio < 0.75:
        return "High"
    if ratio < 0.90:
        return "Medium"
    return "Low"


def build_assessment(profile):
    gaps = []
    strengths = []
    viz_rows = []

    for feature, reference in BENCHMARKS.items():
        current = float(profile[feature])
        ratio = 1.0 if reference == 0 else current / float(reference)
        readiness_pct = max(0.0, min(ratio * 100.0, 120.0))

        viz_rows.append(
            {
                "Area": LABELS[feature],
                "Readiness %": readiness_pct,
                "Target": 100.0,
            }
        )

        if current < float(reference):
            gaps.append(
                {
                    "feature": feature,
                    "area": LABELS[feature],
                    "current": profile[feature],
                    "reference": reference,
                    "priority": gap_priority(current, reference),
                    "recommendation": RECOMMENDATIONS[feature],
                    "ratio": ratio,
                }
            )
        else:
            strengths.append(LABELS[feature])

    order = {"High": 0, "Medium": 1, "Low": 2}
    gaps = sorted(gaps, key=lambda x: (order[x["priority"]], x["ratio"]))
    return gaps, strengths, pd.DataFrame(viz_rows)


def build_pdf(result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"{APP_NAME} Placement Readiness Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CareerReadyTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#172033"),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "CareerReadySubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#667085"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#172033"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#344054"),
    )

    story = [
        Paragraph(APP_NAME, title_style),
        Paragraph(APP_SUBTITLE, subtitle_style),
    ]

    profile_data = [
        ["Student Name", result["student_name"]],
        ["Register Number", result["register_number"]],
        ["Mobile", result["mobile"] or "Not provided"],
        ["Course", result["course"]],
    ]
    profile_table = Table(profile_data, colWidths=[42 * mm, 125 * mm])
    profile_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F6FA")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#344054")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D0D5DD")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([profile_table, Spacer(1, 9)])

    summary_data = [
        ["Placement Result", result["result_label"]],
        ["Estimated Placement Probability", f'{result["probability_pct"]:.1f}%'],
        ["Skill Gaps Identified", str(len(result["gaps"]))],
    ]
    summary_table = Table(summary_data, colWidths=[60 * mm, 107 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEF4FF")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C7D7F4")),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    story.extend([summary_table, Spacer(1, 10)])

    story.append(Paragraph("Skill Gap Analysis", heading_style))
    if result["gaps"]:
        gap_data = [["Area", "Current", "Reference", "Priority", "Recommendation"]]
        for gap in result["gaps"]:
            gap_data.append(
                [
                    gap["area"],
                    str(gap["current"]),
                    str(gap["reference"]),
                    gap["priority"],
                    Paragraph(gap["recommendation"], body_style),
                ]
            )
        gap_table = Table(
            gap_data,
            colWidths=[33 * mm, 20 * mm, 22 * mm, 20 * mm, 72 * mm],
            repeatRows=1,
        )
        gap_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1457D9")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D0D5DD")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(gap_table)
    else:
        story.append(Paragraph("No major gap was detected against the dataset-derived reference profile.", body_style))

    story.append(Paragraph("Priority Action Plan", heading_style))
    if result["gaps"]:
        for idx, gap in enumerate(result["gaps"], start=1):
            story.append(
                Paragraph(
                    f'<b>{idx}. {gap["priority"]} Priority — {gap["area"]}</b><br/>{gap["recommendation"]}',
                    body_style,
                )
            )
            story.append(Spacer(1, 4))
    else:
        story.append(
            Paragraph(
                "Maintain your current preparation and continue practicing aptitude, technical and interview skills.",
                body_style,
            )
        )

    story.extend(
        [
            Spacer(1, 12),
            Paragraph(
                "Note: This report is an educational decision-support output generated from the selected public placement dataset. "
                "It is not a guarantee of employment and the reference values are not universal employer cut-offs.",
                subtitle_style,
            ),
        ]
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def app_header():
    st.markdown(
        f"""
        <div class="cr-header">
            <div class="cr-logo">🎓</div>
            <div>
                <div class="cr-name">{APP_NAME}</div>
                <div class="cr-subtitle">{APP_SUBTITLE}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.session_state.setdefault("screen", "assessment")
st.session_state.setdefault("result", None)

try:
    model = train_model()
except Exception as exc:
    app_header()
    st.error("The prediction service could not be initialized. Please refresh the page.")
    with st.expander("Technical details"):
        st.code(str(exc))
    st.stop()


if st.session_state["screen"] == "assessment":
    st.markdown(
        """
        <style>
        div[data-testid="stButton"] {
            position: sticky;
            bottom: 12px;
            z-index: 999;
            background: rgba(247,249,252,.96);
            padding: .75rem 0 .2rem;
            backdrop-filter: blur(8px);
        }
        div[data-testid="stButton"] button {
            min-height: 48px;
            font-weight: 750;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    app_header()

    st.markdown('<div class="page-title">Check your placement readiness</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-copy">Enter your current academic and employability details. CareerReady will estimate placement readiness and highlight the areas you can improve next.</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.subheader("Student Profile")
        p1, p2 = st.columns(2)
        with p1:
            student_name = st.text_input(
                "Student Name *",
                value=None,
                placeholder="Enter full name",
            )
            register_number = st.text_input(
                "Register Number *",
                value=None,
                placeholder="Enter register / roll number",
            )
        with p2:
            mobile = st.text_input(
                "Mobile Number (Optional)",
                value=None,
                placeholder="10-digit mobile number",
            )
            course = st.selectbox(
                "Course *",
                COURSES,
                index=None,
                placeholder="Select course",
            )

        st.caption("Course is used in the student report only; it is not an input feature of the current ML model.")

    with st.container(border=True):
        st.subheader("Academic & Employability Profile")
        a1, a2 = st.columns(2)

        with a1:
            cgpa = st.number_input(
                "CGPA *",
                min_value=6.5,
                max_value=9.1,
                value=None,
                step=0.1,
                placeholder="6.5 - 9.1",
            )
            ssc_marks = st.number_input(
                "SSC Marks *",
                min_value=55,
                max_value=90,
                value=None,
                step=1,
                placeholder="55 - 90",
            )
            hsc_marks = st.number_input(
                "HSC Marks *",
                min_value=57,
                max_value=88,
                value=None,
                step=1,
                placeholder="57 - 88",
            )
            internships = st.number_input(
                "Internships Completed *",
                min_value=0,
                max_value=2,
                value=None,
                step=1,
                placeholder="0 - 2",
            )
            projects = st.number_input(
                "Projects Completed *",
                min_value=0,
                max_value=3,
                value=None,
                step=1,
                placeholder="0 - 3",
            )

        with a2:
            workshops = st.number_input(
                "Workshops / Certifications *",
                min_value=0,
                max_value=3,
                value=None,
                step=1,
                placeholder="0 - 3",
            )
            aptitude = st.number_input(
                "Aptitude Test Score *",
                min_value=60,
                max_value=90,
                value=None,
                step=1,
                placeholder="60 - 90",
            )
            soft_skills = st.number_input(
                "Soft Skills Rating *",
                min_value=3.0,
                max_value=4.8,
                value=None,
                step=0.1,
                placeholder="3.0 - 4.8",
            )
            extracurricular = st.selectbox(
                "Extracurricular Activities *",
                ["Yes", "No"],
                index=None,
                placeholder="Select",
            )
            placement_training = st.selectbox(
                "Placement Training *",
                ["Yes", "No"],
                index=None,
                placeholder="Select",
            )

    mobile_clean = re.sub(r"\D", "", mobile or "")
    mobile_valid = (not mobile) or (10 <= len(mobile_clean) <= 15)

    required_values = [
        student_name,
        register_number,
        course,
        cgpa,
        ssc_marks,
        hsc_marks,
        internships,
        projects,
        workshops,
        aptitude,
        soft_skills,
        extracurricular,
        placement_training,
    ]
    form_complete = all(value is not None and value != "" for value in required_values)
    can_analyze = form_complete and mobile_valid

    if mobile and not mobile_valid:
        st.warning("Enter a valid mobile number, or leave the mobile field empty.")

    analyze = st.button(
        "Analyze My Placement Readiness",
        type="primary",
        use_container_width=True,
        disabled=not can_analyze,
    )

    if analyze:
        profile = {
            "CGPA": float(cgpa),
            "Internships": int(internships),
            "Projects": int(projects),
            "Workshops/Certifications": int(workshops),
            "AptitudeTestScore": int(aptitude),
            "SoftSkillsRating": float(soft_skills),
            "ExtracurricularActivities": yes_no(extracurricular),
            "PlacementTraining": yes_no(placement_training),
            "SSC_Marks": int(ssc_marks),
            "HSC_Marks": int(hsc_marks),
        }

        input_df = pd.DataFrame([profile], columns=FEATURES)
        probability = float(model.predict_proba(input_df)[0, 1])
        prediction = int(model.predict(input_df)[0])
        gaps, strengths, viz_df = build_assessment(profile)

        st.session_state["result"] = {
            "student_name": student_name.strip(),
            "register_number": register_number.strip(),
            "mobile": mobile_clean,
            "course": course,
            "profile": profile,
            "probability": probability,
            "probability_pct": probability * 100,
            "prediction": prediction,
            "result_label": "Placement Ready" if prediction == 1 else "Needs Improvement",
            "gaps": gaps,
            "strengths": strengths,
            "viz_df": viz_df.to_dict("records"),
        }
        st.session_state["screen"] = "report"
        st.rerun()

else:
    result = st.session_state["result"]

    app_header()

    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown('<div class="page-title">Your placement readiness report</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="page-copy">Prepared for <b>{result["student_name"]}</b> · {result["register_number"]}</div>',
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("← Edit details", use_container_width=True):
            st.session_state["screen"] = "assessment"
            st.rerun()

    st.markdown(
        f"""
        <div class="summary-card">
            <div class="status-label">Placement readiness result</div>
            <div class="status-value">{result["result_label"]}</div>
            <div class="note" style="margin-top:.45rem">
                Estimated placement probability: <b>{result["probability_pct"]:.1f}%</b>
                &nbsp; · &nbsp; Skill gaps identified: <b>{len(result["gaps"])}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Probability", f'{result["probability_pct"]:.1f}%')
    m2.metric("Strength Areas", len(result["strengths"]))
    m3.metric("Improvement Areas", len(result["gaps"]))

    st.progress(min(max(result["probability"], 0.0), 1.0))

    st.subheader("Profile Readiness Overview")
    st.caption("Each bar shows your current value as a percentage of the placed-student reference for that area. 100% represents the reference level.")

    viz_df = pd.DataFrame(result["viz_df"])
    base = alt.Chart(viz_df).encode(
        y=alt.Y("Area:N", sort="-x", title=None),
    )
    bars = base.mark_bar(cornerRadiusEnd=5, color="#4F7DD9").encode(
        x=alt.X("Readiness %:Q", scale=alt.Scale(domain=[0, 120]), title="Readiness against reference (%)"),
        tooltip=["Area", alt.Tooltip("Readiness %:Q", format=".0f")],
    )
    target = alt.Chart(pd.DataFrame({"Target": [100]})).mark_rule(
        color="#D92D20",
        strokeDash=[4, 4],
        size=2,
    ).encode(x="Target:Q")
    st.altair_chart((bars + target).properties(height=300), use_container_width=True)

    if result["strengths"]:
        st.subheader("Your Strengths")
        chips = "".join(
            f'<span class="strength-chip">✓ {item}</span>'
            for item in result["strengths"]
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.subheader("Skill Gap Analysis")
    if result["gaps"]:
        gaps = result["gaps"]
        for idx in range(0, len(gaps), 2):
            cols = st.columns(2)
            for offset, col in enumerate(cols):
                gap_index = idx + offset
                if gap_index >= len(gaps):
                    continue
                gap = gaps[gap_index]
                priority_class = gap["priority"].lower()
                with col:
                    st.markdown(
                        f"""
                        <div class="gap-card">
                            <span class="priority priority-{priority_class}">{gap["priority"]} Priority</span>
                            <div class="gap-area">{gap["area"]}</div>
                            <div class="gap-number">{gap["current"]}</div>
                            <div class="gap-reference">Reference: {gap["reference"]}</div>
                            <div class="gap-reco">{gap["recommendation"]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
    else:
        st.success("No major gap was detected against the dataset-derived placed-student reference profile.")

    st.subheader("What You Should Do Next")
    if result["gaps"]:
        for index, gap in enumerate(result["gaps"], start=1):
            st.markdown(
                f"""
                <div class="action-card">
                    <span class="priority priority-{gap["priority"].lower()}">{gap["priority"]} Priority</span>
                    <div class="action-title">{index}. {gap["area"]}</div>
                    <div class="action-copy">{gap["recommendation"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="action-card">
                <div class="action-title">Maintain your preparation</div>
                <div class="action-copy">Continue practicing aptitude, technical skills and interview preparation while maintaining your current academic performance.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pdf_bytes = build_pdf(result)
    safe_reg = re.sub(r"[^A-Za-z0-9_-]+", "_", result["register_number"]).strip("_") or "student"

    st.download_button(
        "Download Placement Readiness Report (PDF)",
        data=pdf_bytes,
        file_name=f"{safe_reg}_CareerReady_Report.pdf",
        mime="application/pdf",
        type="primary",
        use_container_width=True,
    )

    st.markdown(
        '<div class="note" style="margin-top:.75rem">This is an educational decision-support result based on the selected public placement dataset. It is not a guarantee of employment, and the reference values are not universal employer cut-offs.</div>',
        unsafe_allow_html=True,
    )
