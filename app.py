from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title="Jaya Jaya Institut - Early Warning System",
    page_icon="🎓",
    layout="wide",
)

RISK_BANDS = {
    "Rendah": (0.0, 0.35),
    "Sedang": (0.35, 0.65),
    "Tinggi": (0.65, 1.01),
}

DISPLAY_NAMES = {
    "Course": "Program Studi",
    "Age_at_enrollment": "Usia saat Masuk",
    "Admission_grade": "Nilai Seleksi Masuk",
    "Previous_qualification_grade": "Nilai Pendidikan Sebelumnya",
    "Tuition_fees_up_to_date": "Biaya Kuliah Up to Date",
    "Debtor": "Memiliki Tunggakan",
    "Scholarship_holder": "Penerima Beasiswa",
    "Gender": "Jenis Kelamin",
    "Curricular_units_1st_sem_approved": "MK Lulus Semester 1",
    "Curricular_units_1st_sem_grade": "Rata-rata Nilai Semester 1",
    "Curricular_units_2nd_sem_approved": "MK Lulus Semester 2",
    "Curricular_units_2nd_sem_grade": "Rata-rata Nilai Semester 2",
    "Daytime_evening_attendance": "Kelas Siang/Malam",
    "Marital_status": "Status Pernikahan",
    "Application_mode": "Mode Pendaftaran",
    "Application_order": "Urutan Pilihan",
    "Previous_qualification": "Pendidikan Sebelumnya",
    "Nacionality": "Kewarganegaraan",
    "Mothers_qualification": "Pendidikan Ibu",
    "Fathers_qualification": "Pendidikan Ayah",
    "Mothers_occupation": "Pekerjaan Ibu",
    "Fathers_occupation": "Pekerjaan Ayah",
    "Displaced": "Status Displaced",
    "Educational_special_needs": "Kebutuhan Khusus",
    "International": "Mahasiswa Internasional",
    "Curricular_units_1st_sem_credited": "SKS Kredit Semester 1",
    "Curricular_units_1st_sem_enrolled": "MK Diambil Semester 1",
    "Curricular_units_1st_sem_evaluations": "Jumlah Evaluasi Semester 1",
    "Curricular_units_1st_sem_without_evaluations": "MK Tanpa Evaluasi Semester 1",
    "Curricular_units_2nd_sem_credited": "SKS Kredit Semester 2",
    "Curricular_units_2nd_sem_enrolled": "MK Diambil Semester 2",
    "Curricular_units_2nd_sem_evaluations": "Jumlah Evaluasi Semester 2",
    "Curricular_units_2nd_sem_without_evaluations": "MK Tanpa Evaluasi Semester 2",
    "Unemployment_rate": "Tingkat Pengangguran",
    "Inflation_rate": "Tingkat Inflasi",
    "GDP": "GDP",
}

QUICK_FEATURES = [
    "Course",
    "Age_at_enrollment",
    "Admission_grade",
    "Previous_qualification_grade",
    "Tuition_fees_up_to_date",
    "Debtor",
    "Scholarship_holder",
    "Gender",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
]

SECTIONS = {
    "Profil & Administrasi": [
        "Marital_status",
        "Application_mode",
        "Application_order",
        "Course",
        "Daytime_evening_attendance",
        "Previous_qualification",
        "Previous_qualification_grade",
        "Nacionality",
        "Admission_grade",
        "Displaced",
        "Educational_special_needs",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Gender",
        "Scholarship_holder",
        "Age_at_enrollment",
        "International",
    ],
    "Latar Belakang Keluarga": [
        "Mothers_qualification",
        "Fathers_qualification",
        "Mothers_occupation",
        "Fathers_occupation",
    ],
    "Performa Semester 1": [
        "Curricular_units_1st_sem_credited",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_1st_sem_without_evaluations",
    ],
    "Performa Semester 2": [
        "Curricular_units_2nd_sem_credited",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
        "Curricular_units_2nd_sem_without_evaluations",
    ],
    "Kondisi Makro Ekonomi": [
        "Unemployment_rate",
        "Inflation_rate",
        "GDP",
    ],
}


@st.cache_resource
def load_pipeline():
    return joblib.load(MODEL_DIR / "dropout_pipeline.joblib")


@st.cache_data
def load_metadata() -> dict[str, Any]:
    with open(MODEL_DIR / "feature_metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict[str, Any]:
    with open(MODEL_DIR / "model_metrics.json", "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_sample_template() -> bytes:
    return (DATA_DIR / "sample_input.csv").read_bytes()


@st.cache_data
def load_sample_template_columns() -> list[str]:
    return pd.read_csv(DATA_DIR / "sample_input.csv").columns.tolist()


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
        .hero-card {
            padding: 1.25rem 1.35rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #eff6ff 0%, #f8fafc 100%);
            border: 1px solid #dbeafe;
            margin-bottom: 1rem;
        }
        .mini-card {
            padding: 0.95rem 1rem;
            border-radius: 16px;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            min-height: 110px;
        }
        .section-note {
            padding: 0.9rem 1rem;
            border-radius: 14px;
            background: #f9fafb;
            border: 1px dashed #d1d5db;
            margin-bottom: 0.8rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_default_row(metadata: dict[str, Any]) -> dict[str, Any]:
    return {feature: meta["default"] for feature, meta in metadata.items()}


def risk_level(probability: float) -> str:
    for label, (lower, upper) in RISK_BANDS.items():
        if lower <= probability < upper:
            return label
    return "Tinggi"


def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature.replace("_", " "))


def format_binary_value(feature: str, value: Any) -> str:
    if feature == "Gender":
        return "Laki-laki" if int(value) == 1 else "Perempuan"
    if feature == "Daytime_evening_attendance":
        return "Siang" if int(value) == 1 else "Malam"
    return "Ya" if int(value) == 1 else "Tidak"


def format_feature_value(feature: str, value: Any, metadata: dict[str, Any]) -> str:
    meta = metadata.get(feature, {})
    if pd.isna(value):
        return "-"
    if meta.get("kind") == "binary":
        return format_binary_value(feature, value)
    if meta.get("kind") == "categorical":
        labels = meta.get("labels", {})
        if str(value) in labels:
            return f"{value} - {labels[str(value)]}"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def risk_notes(row: pd.Series, probability: float) -> list[str]:
    notes: list[str] = []
    if row.get("Tuition_fees_up_to_date", 1) == 0:
        notes.append("Biaya kuliah belum up to date.")
    if row.get("Debtor", 0) == 1:
        notes.append("Mahasiswa memiliki tunggakan atau status debtor.")
    if row.get("Scholarship_holder", 0) == 0:
        notes.append("Mahasiswa tidak menerima beasiswa.")
    if row.get("Curricular_units_1st_sem_approved", 0) <= 3:
        notes.append("Jumlah mata kuliah lulus di semester 1 tergolong rendah.")
    if row.get("Curricular_units_2nd_sem_approved", 0) <= 3:
        notes.append("Jumlah mata kuliah lulus di semester 2 tergolong rendah.")
    if row.get("Curricular_units_2nd_sem_grade", 0) < 11:
        notes.append("Rata-rata nilai semester 2 masih rendah.")
    if row.get("Age_at_enrollment", 0) >= 25:
        notes.append("Usia saat masuk relatif lebih tinggi dibanding mayoritas mahasiswa.")
    if probability >= 0.65 and not notes:
        notes.append("Probabilitas model tinggi walaupun tidak ada satu indikator tunggal yang sangat dominan.")
    return notes


def recommended_actions(row: pd.Series, probability: float) -> list[str]:
    actions: list[str] = []
    if row.get("Tuition_fees_up_to_date", 1) == 0 or row.get("Debtor", 0) == 1:
        actions.append("Prioritaskan konseling finansial dan cek opsi keringanan atau skema pembayaran.")
    if row.get("Curricular_units_1st_sem_approved", 0) <= 3 or row.get("Curricular_units_2nd_sem_approved", 0) <= 3:
        actions.append("Jadwalkan bimbingan akademik untuk menyusun target studi jangka pendek yang realistis.")
    if row.get("Curricular_units_1st_sem_grade", 0) < 11 or row.get("Curricular_units_2nd_sem_grade", 0) < 11:
        actions.append("Arahkan ke program remedial atau pendampingan belajar pada mata kuliah inti.")
    if row.get("Scholarship_holder", 0) == 0 and probability >= 0.5:
        actions.append("Tinjau peluang beasiswa, bantuan akademik, atau dukungan administrasi tambahan.")
    if row.get("Age_at_enrollment", 0) >= 25:
        actions.append("Lakukan pendekatan yang lebih personal karena kebutuhan mahasiswa non-tradisional bisa berbeda.")
    if not actions:
        actions.append("Lanjutkan monitoring rutin. Mahasiswa belum menunjukkan sinyal risiko yang kuat.")
    return actions[:4]


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 35], "color": "#d1fae5"},
                    {"range": [35, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"width": 4}, "thickness": 0.75, "value": 45},
            },
            title={"text": "Probabilitas Dropout"},
        )
    )
    fig.update_layout(height=300, margin=dict(l=25, r=25, t=60, b=20))
    return fig


def risk_badge(level: str) -> tuple[str, str]:
    if level == "Tinggi":
        return "Risiko Tinggi", "error"
    if level == "Sedang":
        return "Risiko Sedang", "warning"
    return "Risiko Rendah", "success"


def predict_single(pipeline, row: dict[str, Any], metadata: dict[str, Any], metrics: dict[str, Any]) -> pd.DataFrame:
    ordered_features = list(metadata.keys())
    input_df = pd.DataFrame([{feature: row[feature] for feature in ordered_features}])
    proba = float(pipeline.predict_proba(input_df)[0, 1])
    threshold = float(metrics.get("threshold", 0.5))
    prediction = int(proba >= threshold)
    input_df["dropout_probability"] = proba
    input_df["prediction_label"] = "Dropout Risk" if prediction == 1 else "Non-Dropout Risk"
    input_df["risk_level"] = risk_level(proba)
    return input_df


def predict_batch(pipeline, uploaded_df: pd.DataFrame, metadata: dict[str, Any], metrics: dict[str, Any]) -> pd.DataFrame:
    required_columns = list(metadata.keys())
    df = uploaded_df.copy()
    if "Status" in df.columns:
        df = df.drop(columns=["Status"])
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Kolom berikut belum ada di file upload: {missing_columns}")
    df = df[required_columns]
    proba = pipeline.predict_proba(df)[:, 1]
    threshold = float(metrics.get("threshold", 0.5))
    pred = (proba >= threshold).astype(int)
    output = df.copy()
    output["dropout_probability"] = proba
    output["prediction_label"] = ["Dropout Risk" if x == 1 else "Non-Dropout Risk" for x in pred]
    output["risk_level"] = [risk_level(x) for x in proba]
    return output.sort_values("dropout_probability", ascending=False).reset_index(drop=True)


def validate_batch_columns(uploaded_df: pd.DataFrame, metadata: dict[str, Any]) -> tuple[list[str], list[str]]:
    required_columns = list(metadata.keys())
    available = set(uploaded_df.columns)
    missing = [col for col in required_columns if col not in available]
    extras = [col for col in uploaded_df.columns if col not in required_columns and col != "Status"]
    return missing, extras


def render_input_widget(feature: str, meta: dict[str, Any], key_prefix: str) -> Any:
    description = meta.get("description", "")
    label = display_name(feature)
    widget_key = f"{key_prefix}_{feature}"

    if meta["kind"] == "binary":
        options = meta.get("options", [0, 1])
        index = options.index(meta["default"]) if meta["default"] in options else 0
        selected = st.selectbox(
            label,
            options=options,
            index=index,
            format_func=lambda x: format_binary_value(feature, x),
            help=description,
            key=widget_key,
        )
        return int(selected)

    if meta["kind"] == "categorical":
        options = meta.get("options", [])
        labels = meta.get("labels", {})
        try:
            index = options.index(meta["default"])
        except ValueError:
            index = 0
        selected = st.selectbox(
            label,
            options=options,
            index=index,
            format_func=lambda x: f"{x} - {labels.get(str(x), '')}".strip(" -"),
            help=description,
            key=widget_key,
        )
        if isinstance(selected, float) and selected.is_integer():
            return int(selected)
        return selected

    min_value = float(meta.get("min", 0.0))
    max_value = float(meta.get("max", 100.0))
    default = float(meta.get("default", 0.0))
    step = 1.0 if abs(default - round(default)) < 1e-9 else 0.1
    value = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        help=description,
        key=widget_key,
    )
    return int(value) if step == 1.0 else float(value)


def apply_quick_preset(name: str, metadata: dict[str, Any]) -> None:
    base = build_default_row(metadata)
    presets: dict[str, dict[str, Any]] = {
        "default": {
            "Course": base["Course"],
            "Age_at_enrollment": int(round(base["Age_at_enrollment"])),
            "Admission_grade": float(base["Admission_grade"]),
            "Previous_qualification_grade": float(base["Previous_qualification_grade"]),
            "Tuition_fees_up_to_date": 1,
            "Debtor": 0,
            "Scholarship_holder": 0,
            "Gender": 0,
            "Curricular_units_1st_sem_approved": 4,
            "Curricular_units_1st_sem_grade": 12.0,
            "Curricular_units_2nd_sem_approved": 4,
            "Curricular_units_2nd_sem_grade": 12.0,
        },
        "low": {
            "Course": 9500,
            "Age_at_enrollment": 19,
            "Admission_grade": 135.0,
            "Previous_qualification_grade": 140.0,
            "Tuition_fees_up_to_date": 1,
            "Debtor": 0,
            "Scholarship_holder": 1,
            "Gender": 0,
            "Curricular_units_1st_sem_approved": 7,
            "Curricular_units_1st_sem_grade": 14.0,
            "Curricular_units_2nd_sem_approved": 7,
            "Curricular_units_2nd_sem_grade": 14.0,
        },
        "high": {
            "Course": 9991,
            "Age_at_enrollment": 29,
            "Admission_grade": 105.0,
            "Previous_qualification_grade": 105.0,
            "Tuition_fees_up_to_date": 0,
            "Debtor": 1,
            "Scholarship_holder": 0,
            "Gender": 1,
            "Curricular_units_1st_sem_approved": 1,
            "Curricular_units_1st_sem_grade": 9.5,
            "Curricular_units_2nd_sem_approved": 0,
            "Curricular_units_2nd_sem_grade": 8.0,
        },
    }
    for feature, value in presets[name].items():
        st.session_state[f"quick_{feature}"] = value


def quick_assessment_form(metadata: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    row = build_default_row(metadata)
    course_labels = metadata["Course"].get("labels", {})
    course_options = metadata["Course"].get("options", [])

    st.markdown("### Prediksi cepat")
    st.caption(
        "Isi faktor-faktor utama di bawah ini. Fitur lain tetap memakai nilai default model, kecuali diubah pada bagian lanjutan."
    )

    with st.form("single_prediction_form"):
        st.markdown("#### Profil singkat")
        col1, col2 = st.columns(2)
        with col1:
            row["Course"] = st.selectbox(
                display_name("Course"),
                options=course_options,
                index=course_options.index(st.session_state.get("quick_Course", metadata["Course"]["default"])),
                format_func=lambda x: f"{course_labels.get(str(x), x)}",
                key="quick_Course",
            )
            row["Age_at_enrollment"] = st.number_input(
                display_name("Age_at_enrollment"),
                min_value=int(metadata["Age_at_enrollment"]["min"]),
                max_value=int(metadata["Age_at_enrollment"]["max"]),
                value=int(st.session_state.get("quick_Age_at_enrollment", round(metadata["Age_at_enrollment"]["default"]))),
                step=1,
                key="quick_Age_at_enrollment",
            )
            row["Gender"] = st.selectbox(
                display_name("Gender"),
                [0, 1],
                index=[0, 1].index(int(st.session_state.get("quick_Gender", metadata["Gender"]["default"]))),
                format_func=lambda x: format_binary_value("Gender", x),
                key="quick_Gender",
            )
            row["Previous_qualification_grade"] = st.number_input(
                display_name("Previous_qualification_grade"),
                min_value=float(metadata["Previous_qualification_grade"]["min"]),
                max_value=float(metadata["Previous_qualification_grade"]["max"]),
                value=float(st.session_state.get("quick_Previous_qualification_grade", metadata["Previous_qualification_grade"]["default"])),
                step=0.1,
                key="quick_Previous_qualification_grade",
            )
        with col2:
            row["Admission_grade"] = st.number_input(
                display_name("Admission_grade"),
                min_value=float(metadata["Admission_grade"]["min"]),
                max_value=float(metadata["Admission_grade"]["max"]),
                value=float(st.session_state.get("quick_Admission_grade", metadata["Admission_grade"]["default"])),
                step=0.1,
                key="quick_Admission_grade",
            )
            row["Tuition_fees_up_to_date"] = st.selectbox(
                display_name("Tuition_fees_up_to_date"),
                [1, 0],
                index=[1, 0].index(int(st.session_state.get("quick_Tuition_fees_up_to_date", 1))),
                format_func=lambda x: "Ya" if x == 1 else "Tidak",
                key="quick_Tuition_fees_up_to_date",
            )
            row["Debtor"] = st.selectbox(
                display_name("Debtor"),
                [0, 1],
                index=[0, 1].index(int(st.session_state.get("quick_Debtor", 0))),
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                key="quick_Debtor",
            )
            row["Scholarship_holder"] = st.selectbox(
                display_name("Scholarship_holder"),
                [0, 1],
                index=[0, 1].index(int(st.session_state.get("quick_Scholarship_holder", 0))),
                format_func=lambda x: "Tidak" if x == 0 else "Ya",
                key="quick_Scholarship_holder",
            )

        st.markdown("#### Performa akademik")
        ac1, ac2 = st.columns(2)
        with ac1:
            row["Curricular_units_1st_sem_approved"] = st.number_input(
                display_name("Curricular_units_1st_sem_approved"),
                min_value=0,
                max_value=26,
                value=int(st.session_state.get("quick_Curricular_units_1st_sem_approved", 4)),
                step=1,
                key="quick_Curricular_units_1st_sem_approved",
            )
            row["Curricular_units_1st_sem_grade"] = st.number_input(
                display_name("Curricular_units_1st_sem_grade"),
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.get("quick_Curricular_units_1st_sem_grade", 12.0)),
                step=0.1,
                key="quick_Curricular_units_1st_sem_grade",
            )
        with ac2:
            row["Curricular_units_2nd_sem_approved"] = st.number_input(
                display_name("Curricular_units_2nd_sem_approved"),
                min_value=0,
                max_value=20,
                value=int(st.session_state.get("quick_Curricular_units_2nd_sem_approved", 4)),
                step=1,
                key="quick_Curricular_units_2nd_sem_approved",
            )
            row["Curricular_units_2nd_sem_grade"] = st.number_input(
                display_name("Curricular_units_2nd_sem_grade"),
                min_value=0.0,
                max_value=20.0,
                value=float(st.session_state.get("quick_Curricular_units_2nd_sem_grade", 12.0)),
                step=0.1,
                key="quick_Curricular_units_2nd_sem_grade",
            )

        with st.expander("Pengaturan lanjutan untuk seluruh fitur"):
            st.caption("Bagian ini opsional. Gunakan kalau ingin menyesuaikan semua fitur model secara penuh.")
            for section_name, features in SECTIONS.items():
                st.markdown(f"**{section_name}**")
                cols = st.columns(2)
                filtered = [feature for feature in features if feature not in QUICK_FEATURES]
                for idx, feature in enumerate(filtered):
                    with cols[idx % 2]:
                        row[feature] = render_input_widget(feature, metadata[feature], "advanced")
                st.divider()

        submitted = st.form_submit_button("Analisis Risiko Dropout", use_container_width=True, type="primary")

    return row, submitted


def show_prediction_result(result_df: pd.DataFrame, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    probability = float(result_df["dropout_probability"].iloc[0])
    predicted_label = result_df["prediction_label"].iloc[0]
    level = result_df["risk_level"].iloc[0]
    badge_text, badge_kind = risk_badge(level)

    st.markdown("### Hasil analisis")
    left, right = st.columns([1.1, 1.2])
    with left:
        st.plotly_chart(make_gauge(probability), use_container_width=True)
    with right:
        m1, m2, m3 = st.columns(3)
        m1.metric("Probabilitas Dropout", f"{probability:.1%}")
        m2.metric("Prediksi Model", "Berisiko" if predicted_label == "Dropout Risk" else "Tidak Berisiko")
        m3.metric("Kategori", badge_text)

        if badge_kind == "error":
            st.error("Mahasiswa ini masuk kelompok prioritas tinggi untuk ditindaklanjuti.")
        elif badge_kind == "warning":
            st.warning("Mahasiswa ini perlu dipantau lebih dekat karena sinyal risikonya menengah.")
        else:
            st.success("Mahasiswa ini belum menunjukkan sinyal risiko dropout yang kuat.")

        st.markdown(
            f"Threshold operasional model saat ini adalah **{metrics.get('threshold', 0.5):.2f}**. "
            "Skor di atas threshold akan ditandai sebagai berisiko dropout."
        )

    notes = risk_notes(result_df.iloc[0], probability)
    actions = recommended_actions(result_df.iloc[0], probability)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Faktor yang paling patut diperhatikan")
        if notes:
            for note in notes:
                st.write(f"- {note}")
        else:
            st.write("- Tidak ada indikator tunggal yang sangat menonjol dari input yang diberikan.")
    with col_b:
        st.markdown("#### Rekomendasi tindak lanjut")
        for action in actions:
            st.write(f"- {action}")

    preview_cols = [
        "Course",
        "Age_at_enrollment",
        "Gender",
        "Tuition_fees_up_to_date",
        "Debtor",
        "Scholarship_holder",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
    ]
    preview = []
    row = result_df.iloc[0]
    for feature in preview_cols:
        preview.append(
            {
                "Fitur": display_name(feature),
                "Nilai": format_feature_value(feature, row[feature], metadata),
            }
        )
    with st.expander("Lihat ringkasan input yang dianalisis"):
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)


def render_quick_preset_controls(metadata: dict[str, Any]) -> None:
    st.markdown("#### Preset cepat")
    st.caption("Pakai preset kalau ingin mencoba contoh profil mahasiswa tanpa mengisi semuanya dari nol.")
    p1, p2, p3 = st.columns(3)
    with p1:
        if st.button("Reset ke default", use_container_width=True):
            apply_quick_preset("default", metadata)
            st.rerun()
    with p2:
        if st.button("Contoh risiko rendah", use_container_width=True):
            apply_quick_preset("low", metadata)
            st.rerun()
    with p3:
        if st.button("Contoh risiko tinggi", use_container_width=True):
            apply_quick_preset("high", metadata)
            st.rerun()


def render_batch_section(pipeline, metadata: dict[str, Any], metrics: dict[str, Any]) -> None:
    st.markdown("### Prediksi batch")
    st.caption(
        "Upload file CSV dengan struktur kolom yang sama seperti data training. Kolom `Status` boleh ada atau tidak."
    )

    box1, box2 = st.columns([1, 1.3])
    with box1:
        st.markdown(
            """
            <div class="section-note">
            <b>Yang perlu disiapkan</b><br>
            1. File harus format CSV.<br>
            2. Nama kolom harus sama dengan template.<br>
            3. Nilai kategorikal tetap gunakan kode asli dataset training.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download template batch",
            data=load_sample_template(),
            file_name="sample_input.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with box2:
        st.write("Kolom wajib:")
        st.code(", ".join(load_sample_template_columns()[:8]) + ", ...", language=None)

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is None:
        return

    try:
        batch_df = pd.read_csv(uploaded_file, sep=None, engine="python")
    except Exception as exc:
        st.error(f"File tidak bisa dibaca: {exc}")
        return

    missing_columns, extra_columns = validate_batch_columns(batch_df, metadata)
    preview_col1, preview_col2 = st.columns([1.2, 1])
    with preview_col1:
        st.markdown("#### Preview data upload")
        st.dataframe(batch_df.head(10), use_container_width=True)
    with preview_col2:
        st.markdown("#### Validasi struktur file")
        st.metric("Jumlah baris", len(batch_df))
        st.metric("Kolom terbaca", len(batch_df.columns))
        if missing_columns:
            st.error("Masih ada kolom wajib yang belum lengkap.")
            st.write(missing_columns)
        else:
            st.success("Semua kolom wajib sudah lengkap.")
        if extra_columns:
            st.info("Kolom tambahan akan diabaikan saat prediksi.")
            st.write(extra_columns)

    if missing_columns:
        return

    if st.button("Jalankan prediksi batch", type="primary", use_container_width=True):
        try:
            result = predict_batch(pipeline, batch_df, metadata, metrics)
        except Exception as exc:
            st.error(str(exc))
            return

        total = len(result)
        high = int((result["risk_level"] == "Tinggi").sum())
        medium = int((result["risk_level"] == "Sedang").sum())
        risky = int((result["prediction_label"] == "Dropout Risk").sum())

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Mahasiswa", total)
        s2.metric("Berisiko Dropout", risky)
        s3.metric("Risiko Tinggi", high)
        s4.metric("Rata-rata Probabilitas", f"{result['dropout_probability'].mean():.1%}")

        chart_data = result["risk_level"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"], fill_value=0)
        fig = go.Figure(go.Bar(x=chart_data.index.tolist(), y=chart_data.values.tolist()))
        fig.update_layout(title="Distribusi level risiko", height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        top_cols = [
            "Course",
            "Age_at_enrollment",
            "Gender",
            "Tuition_fees_up_to_date",
            "Debtor",
            "Scholarship_holder",
            "Curricular_units_1st_sem_approved",
            "Curricular_units_2nd_sem_approved",
            "dropout_probability",
            "risk_level",
            "prediction_label",
        ]
        st.markdown("#### Mahasiswa dengan prioritas tertinggi")
        st.dataframe(result[top_cols].head(20), use_container_width=True)

        csv_data = result.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download hasil prediksi batch",
            data=csv_data,
            file_name="batch_prediction_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def render_model_info(metrics: dict[str, Any]) -> None:
    st.markdown("### Informasi model")
    a, b, c, d = st.columns(4)
    a.metric("Model", metrics.get("selected_model", "Unknown"))
    b.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    c.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    d.metric("Precision", f"{metrics.get('precision', 0):.3f}")

    st.markdown(
        """
        **Cara membaca hasil model:**
        - **Probabilitas dropout** menunjukkan seberapa kuat model menilai mahasiswa berada pada kelompok berisiko.
        - **Prediksi model** ditentukan dari threshold operasional 0.45.
        - **Level risiko** dibagi menjadi rendah, sedang, dan tinggi untuk membantu prioritas tindak lanjut.
        - Hasil model dipakai sebagai alat bantu screening awal, bukan keputusan akademik final.
        """
    )

    with st.expander("Metrik evaluasi lengkap"):
        st.json(metrics)


def main() -> None:
    inject_custom_css()
    pipeline = load_pipeline()
    metadata = load_metadata()
    metrics = load_metrics()

    st.markdown(
        """
        <div class="hero-card">
            <h2 style="margin-bottom:0.3rem;">🎓 Jaya Jaya Institut - Early Warning System</h2>
            <p style="margin-bottom:0;">Aplikasi ini membantu mengidentifikasi mahasiswa yang berisiko dropout lebih awal berdasarkan faktor akademik, finansial, dan administratif.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="mini-card"><b>1. Isi data mahasiswa</b><br>Gunakan prediksi cepat untuk satu mahasiswa atau upload CSV untuk banyak mahasiswa.</div>', unsafe_allow_html=True)
    with m2:
        st.markdown('<div class="mini-card"><b>2. Baca hasil risiko</b><br>Lihat probabilitas dropout, level risiko, dan indikator yang paling menonjol.</div>', unsafe_allow_html=True)
    with m3:
        st.markdown('<div class="mini-card"><b>3. Tindak lanjuti</b><br>Gunakan rekomendasi sistem sebagai dasar prioritas intervensi akademik atau finansial.</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("Ringkasan model")
        st.write(f"**Model:** {metrics.get('selected_model', 'Unknown')}")
        st.write(f"**ROC-AUC:** {metrics.get('roc_auc', 0):.3f}")
        st.write(f"**Recall:** {metrics.get('recall', 0):.3f}")
        st.write(f"**Precision:** {metrics.get('precision', 0):.3f}")
        st.write(f"**Threshold:** {metrics.get('threshold', 0.5):.2f}")
        st.divider()
        st.caption("Gunakan template batch kalau ingin memproses banyak mahasiswa sekaligus.")

    tab1, tab2, tab3 = st.tabs(["Prediksi Cepat", "Prediksi Batch", "Informasi Model"])

    with tab1:
        top_left, top_right = st.columns([1.3, 1])
        with top_right:
            render_quick_preset_controls(metadata)
            st.info(
                "Prediksi cepat cocok untuk screening satu mahasiswa. Kalau butuh semua fitur lengkap, buka bagian pengaturan lanjutan."
            )
        with top_left:
            row, submitted = quick_assessment_form(metadata)
        if submitted:
            result = predict_single(pipeline, row, metadata, metrics)
            show_prediction_result(result, metadata, metrics)

    with tab2:
        render_batch_section(pipeline, metadata, metrics)

    with tab3:
        render_model_info(metrics)


if __name__ == "__main__":
    main()
