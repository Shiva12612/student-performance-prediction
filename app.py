import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# PROFESSIONAL UI STYLE
# =========================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main headings */
    h1, h2, h3, h4, h5, h6 {
        color: #172554 !important;
    }

    /* Normal text */
    p, label {
        color: #334155 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    [data-testid="stMetricLabel"] {
        color: #475569 !important;
    }

    [data-testid="stMetricValue"] {
        color: #1d4ed8 !important;
        font-weight: 700;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background-color: #2563eb;
        color: white !important;
        font-weight: 600;
        padding: 10px;
    }

    .stButton > button:hover {
        background-color: #1d4ed8;
        color: white !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172554;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: white !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Alerts */
    .stAlert {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODELS
# =========================================================

regression_model = joblib.load(
    "models/regression_model.pkl"
)

classification_model = joblib.load(
    "models/classification_model.pkl"
)


# =========================================================
# FEATURES
# =========================================================

features = [
    "study_hours",
    "attendance",
    "previous_marks",
    "assignment_score",
    "internal_marks",
    "sleep_hours",
    "internet_usage",
    "extracurricular"
]


# =========================================================
# TITLE
# =========================================================

st.title(
    "🎓 Student Performance Prediction System"
)

st.caption(
    "Machine Learning Based Academic Performance Analysis"
)


st.divider()


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(
    "data/student_data.csv"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🎓 Navigation"
)

page = st.sidebar.radio(
    "Select Page",
    [
        "📊 Dashboard",
        "👤 Individual Prediction",
        "📁 CSV Prediction",
        "📈 Analytics"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "📊 Dashboard":

    st.header(
        "📊 Dashboard Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Students",
            len(df)
        )

    with col2:

        st.metric(
            "Average Final Marks",
            f"{df['final_marks'].mean():.2f}"
        )

    with col3:

        st.metric(
            "Average Attendance",
            f"{df['attendance'].mean():.2f}%"
        )

    with col4:

        excellent_count = (
            df["performance"] == "Excellent"
        ).sum()

        st.metric(
            "Excellent Students",
            excellent_count
        )


    st.divider()


    # -----------------------------------------
    # PERFORMANCE CHARTS
    # -----------------------------------------

    distribution = (
        df["performance"]
        .value_counts()
    )

    # Bar chart and donut chart side-by-side
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:

        st.subheader(
            "📊 Performance Distribution"
        )

        st.bar_chart(
            distribution
        )

    with chart_col2:

        st.subheader(
            "🍩 Performance Share"
        )

        # Colorful donut chart
        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.pie(
            distribution.values,
            labels=distribution.index,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False,
            wedgeprops={
                "width": 0.42,
                "edgecolor": "white",
                "linewidth": 2
            },
            textprops={
                "fontsize": 9
            }
        )

        ax.set_title(
            "Student Performance Share",
            fontsize=12,
            fontweight="bold",
            pad=12
        )

        ax.axis("equal")

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # -----------------------------------------
    # AVERAGE MARKS BY CATEGORY
    # -----------------------------------------

    st.subheader(
        "📈 Average Marks by Category"
    )

    category_marks = (
        df.groupby("performance")[
            "final_marks"
        ].mean()
    )

    st.bar_chart(
        category_marks
    )


    st.divider()


    st.subheader(
        "👨‍🎓 Student Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# INDIVIDUAL PREDICTION
# =========================================================

elif page == "👤 Individual Prediction":

    st.header(
        "👤 Individual Student Prediction"
    )

    st.write(
        "Enter student details below and "
        "predict academic performance."
    )


    col1, col2 = st.columns(2)


    with col1:

        study_hours = st.number_input(
            "📚 Study Hours per Day",
            min_value=0.0,
            max_value=15.0,
            value=6.0,
            step=0.5
        )


        attendance = st.slider(
            "📅 Attendance (%)",
            min_value=0,
            max_value=100,
            value=85
        )


        previous_marks = st.number_input(
            "📝 Previous Marks",
            min_value=0.0,
            max_value=100.0,
            value=72.0
        )


        assignment_score = st.number_input(
            "📋 Assignment Score",
            min_value=0.0,
            max_value=100.0,
            value=80.0
        )


    with col2:

        internal_marks = st.number_input(
            "📖 Internal Marks",
            min_value=0.0,
            max_value=100.0,
            value=75.0
        )


        sleep_hours = st.number_input(
            "😴 Sleep Hours",
            min_value=0.0,
            max_value=15.0,
            value=7.0,
            step=0.5
        )


        internet_usage = st.number_input(
            "🌐 Internet Usage (Hours)",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.5
        )


        extracurricular = st.selectbox(
            "🏆 Extracurricular Activity",
            ["No", "Yes"]
        )


    extracurricular_value = (
        1 if extracurricular == "Yes"
        else 0
    )


    st.divider()


    if st.button(
        "🔮 Predict Performance",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{

            "study_hours": study_hours,

            "attendance": attendance,

            "previous_marks": previous_marks,

            "assignment_score": assignment_score,

            "internal_marks": internal_marks,

            "sleep_hours": sleep_hours,

            "internet_usage": internet_usage,

            "extracurricular": extracurricular_value

        }])


        # -------------------------------------
        # PREDICTION
        # -------------------------------------

        predicted_marks = (
            regression_model
            .predict(input_data)[0]
        )


        predicted_category = (
            classification_model
            .predict(input_data)[0]
        )


        # -------------------------------------
        # DISPLAY RESULTS
        # -------------------------------------

        st.subheader(
            "🎯 Prediction Result"
        )


        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "Predicted Final Marks",
                f"{predicted_marks:.2f} / 100"
            )


        with result_col2:

            st.metric(
                "Performance Category",
                predicted_category
            )


        st.success(
            f"Prediction completed: "
            f"{predicted_category}"
        )


        # -------------------------------------
        # STUDENT SUMMARY
        # -------------------------------------

        st.subheader(
            "📋 Student Input Summary"
        )

        st.dataframe(
            input_data,
            use_container_width=True
        )


# =========================================================
# CSV PREDICTION
# =========================================================

elif page == "📁 CSV Prediction":

    st.header(
        "📁 Batch Student Prediction"
    )

    st.write(
        "Upload a CSV file containing multiple "
        "students and generate predictions."
    )


    st.info(
        "CSV must contain all required feature columns."
    )


    st.code(
        """
student_id,study_hours,attendance,previous_marks,
assignment_score,internal_marks,sleep_hours,
internet_usage,extracurricular
        """
    )


    uploaded_file = st.file_uploader(
        "Upload Student CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        uploaded_df = pd.read_csv(
            uploaded_file
        )


        st.subheader(
            "📄 Uploaded Data"
        )

        st.dataframe(
            uploaded_df,
            use_container_width=True
        )


        missing_columns = [
            col for col in features
            if col not in uploaded_df.columns
        ]


        if missing_columns:

            st.error(
                "Missing columns: "
                + ", ".join(missing_columns)
            )


        else:

            if st.button(
                "🚀 Predict All Students"
            ):

                prediction_input = (
                    uploaded_df[features]
                )


                # Marks prediction
                predicted_marks = (
                    regression_model
                    .predict(prediction_input)
                )


                # Category prediction
                predicted_category = (
                    classification_model
                    .predict(prediction_input)
                )


                result_df = uploaded_df.copy()


                result_df[
                    "predicted_marks"
                ] = np.round(
                    predicted_marks,
                    2
                )


                result_df[
                    "predicted_performance"
                ] = predicted_category


                st.success(
                    "Prediction completed successfully!"
                )


                st.subheader(
                    "🎯 Prediction Results"
                )


                st.dataframe(
                    result_df,
                    use_container_width=True
                )


                # --------------------------------
                # DOWNLOAD RESULT
                # --------------------------------

                csv_data = (
                    result_df
                    .to_csv(index=False)
                    .encode("utf-8")
                )


                st.download_button(
                    label="⬇️ Download Predictions CSV",
                    data=csv_data,
                    file_name="student_predictions.csv",
                    mime="text/csv"
                )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📈 Analytics":

    st.header(
        "📈 Student Performance Analytics"
    )


    # -----------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------

    st.subheader(
        "🔍 Feature Importance"
    )


    importance_file = (
        "models/feature_importance.csv"
    )


    if os.path.exists(
        importance_file
    ):

        importance_df = pd.read_csv(
            importance_file
        )


        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
        )


        st.bar_chart(
            importance_df.set_index(
                "feature"
            )
        )


        st.dataframe(
            importance_df,
            use_container_width=True
        )


    st.divider()


    # -----------------------------------------
    # MODEL PERFORMANCE
    # -----------------------------------------

    st.subheader(
        "🤖 Model Performance"
    )


    metric_col1, metric_col2 = st.columns(2)


    with metric_col1:

        st.info(
            "Linear Regression\n\n"
            "Used for Final Marks Prediction"
        )


        st.metric(
            "R² Score",
            "0.8972"
        )


    with metric_col2:

        st.info(
            "Random Forest\n\n"
            "Used for Performance Classification"
        )


        st.metric(
            "Classification Accuracy",
            "68%"
        )


    st.divider()


    # -----------------------------------------
    # DATA STATISTICS
    # -----------------------------------------

    st.subheader(
        "📊 Dataset Statistics"
    )


    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Student Performance Prediction System | "
    "Machine Learning Academic Project"
)