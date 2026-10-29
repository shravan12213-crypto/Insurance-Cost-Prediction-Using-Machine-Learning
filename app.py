
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Insurance Cost Predictor",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "insurance.csv"
MODEL_DIR = BASE_DIR / "models"

BEST_MODEL_PATH = MODEL_DIR / "insurance_model.joblib"
COMPARISON_PATH = MODEL_DIR / "model_comparison.csv"
RESULTS_PATH = MODEL_DIR / "test_results.csv"


# =========================================================
# 2. LOAD DATA AND TRAINED MODELS
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data
def load_test_results():
    return pd.read_csv(RESULTS_PATH)


@st.cache_resource
def load_best_model():
    return joblib.load(BEST_MODEL_PATH)


df = load_data()
comparison_df = load_comparison()
test_results = load_test_results()
best_model = load_best_model()


# =========================================================
# 3. CALCULATE EVALUATION METRICS
# =========================================================

actual = test_results["actual_charges"]
predicted = test_results["predicted_charges"]

mae = mean_absolute_error(actual, predicted)
r2 = r2_score(actual, predicted)

best_row = comparison_df.loc[
    comparison_df["MAE"].idxmin()
]

best_model_name = best_row["Model"]


# =========================================================
# 4. HEADER
# =========================================================

st.title("Insurance Cost Predictor")

st.markdown(
    """
    Estimate medical insurance charges using machine learning.

    This dashboard compares three regression algorithms and
    provides interactive insurance cost predictions.
    """
)

st.divider()


# =========================================================
# 5. SIDEBAR
# =========================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a section",
    [
        "Dashboard Overview",
        "Model Comparison",
        "Data Visualizations",
        "Insurance Prediction",
        "About the Project"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Project | Python | Scikit-learn"
)


# =========================================================
# PAGE 1: DASHBOARD OVERVIEW
# =========================================================

if page == "Dashboard Overview":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        f"{len(df):,}"
    )

    col2.metric(
        "Mean Absolute Error",
        f"${mae:,.2f}"
    )

    col3.metric(
        "R² Score",
        f"{r2:.4f}"
    )

    st.divider()

    st.subheader("Best Performing Model")

    col1, col2 = st.columns(2)

    with col1:
        st.success(f"🏆 {best_model_name}")

    with col2:
        st.metric(
            "Lowest Test MAE",
            f"${best_row['MAE']:,.2f}"
        )

    st.write(
        "The selected model has the lowest MAE among the "
        "three models evaluated on the test set."
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Dataset shape:**", df.shape)
        st.write("**Features:**", len(df.columns) - 1)
        st.write("**Target:** charges")

    with col2:
        st.write(
            "**Missing values:**",
            int(df.isnull().sum().sum())
        )

        st.write(
            "**Training records:**",
            len(df) - len(test_results)
        )

        st.write(
            "**Testing records:**",
            len(test_results)
        )


# =========================================================
# PAGE 2: MODEL COMPARISON
# =========================================================

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    st.write(
        "Three regression models were trained using the "
        "same training and testing split."
    )

    st.subheader("Performance Summary")

    display_df = comparison_df.copy()

    display_df["MAE"] = display_df["MAE"].map(
        lambda value: f"${value:,.2f}"
    )

    display_df["R2 Score"] = display_df["R2 Score"].map(
        lambda value: f"{value:.4f}"
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        f"🏆 Best model based on test MAE: {best_model_name}"
    )

    st.divider()

    # MAE chart

    st.subheader("Mean Absolute Error (MAE)")

    mae_chart = comparison_df.set_index("Model")[["MAE"]]

    st.bar_chart(mae_chart)

    st.caption(
        "Lower MAE indicates smaller average absolute "
        "prediction errors."
    )

    st.divider()

    # R2 chart

    st.subheader("R² Score")

    r2_chart = comparison_df.set_index("Model")[["R2 Score"]]

    st.bar_chart(r2_chart)

    st.caption(
        "A higher R² generally indicates a better fit "
        "on the test set."
    )

    st.divider()

    st.subheader("Model Interpretation")

    st.markdown(
        """
        **Linear Regression**
        - Fits a linear relationship between features and charges.
        - Simple and easy to interpret.

        **Random Forest**
        - Combines predictions from multiple decision trees.
        - Can capture nonlinear relationships.

        **Gradient Boosting**
        - Builds decision trees sequentially.
        - Each new tree attempts to improve existing predictions.
        """
    )


# =========================================================
# PAGE 3: DATA VISUALIZATIONS
# =========================================================

elif page == "Data Visualizations":

    st.header("Data Visualizations")

    # -----------------------------------------------------
    # GRAPH 1: AGE VS CHARGES
    # -----------------------------------------------------

    st.subheader("1. Age vs. Insurance Charges")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    ax1.scatter(
        df["age"],
        df["charges"],
        alpha=0.6
    )

    ax1.set_xlabel("Age")
    ax1.set_ylabel("Insurance Charges ($)")
    ax1.set_title("Age vs. Insurance Charges")

    ax1.grid(alpha=0.2)

    fig1.tight_layout()

    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "Each point represents one record. "
        "The graph helps explore the relationship between "
        "age and insurance charges."
    )

    st.divider()

    # -----------------------------------------------------
    # GRAPH 2: SMOKING VS AVERAGE CHARGES
    # -----------------------------------------------------

    st.subheader("2. Smoking Status vs. Average Charges")

    average_charges = (
        df.groupby("smoker")["charges"]
        .mean()
        .reindex(["no", "yes"])
    )

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    average_charges.plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_xlabel("Smoking Status")
    ax2.set_ylabel("Average Insurance Charges ($)")
    ax2.set_title("Average Charges by Smoking Status")

    ax2.tick_params(axis="x", rotation=0)
    ax2.grid(axis="y", alpha=0.2)

    fig2.tight_layout()

    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "This chart compares average charges between "
        "smokers and non-smokers in the dataset."
    )

    st.divider()

    # -----------------------------------------------------
    # GRAPH 3: ACTUAL VS PREDICTED CHARGES
    # -----------------------------------------------------

    st.subheader("3. Actual vs. Predicted Charges")

    fig3, ax3 = plt.subplots(figsize=(8, 6))

    ax3.scatter(
        actual,
        predicted,
        alpha=0.6
    )

    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())

    ax3.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax3.set_xlabel("Actual Charges ($)")
    ax3.set_ylabel("Predicted Charges ($)")
    ax3.set_title(
        f"Actual vs. Predicted Charges — {best_model_name}"
    )

    ax3.grid(alpha=0.2)

    fig3.tight_layout()

    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "Points closer to the diagonal line indicate "
        "predictions closer to the actual charges."
    )


# =========================================================
# PAGE 4: INSURANCE PREDICTION
# =========================================================

elif page == "Insurance Prediction":

    st.header("Predict Insurance Costs")

    st.write(
        "Enter the details below to compare predictions "
        "from the available regression models."
    )

    with st.form("insurance_prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=25
            )

            sex = st.selectbox(
                "Sex",
                ["male", "female"]
            )

        with col2:

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=70.0,
                value=24.5,
                step=0.1
            )

            children = st.number_input(
                "Number of Children",
                min_value=0,
                max_value=10,
                value=0
            )

        with col3:

            smoker = st.selectbox(
                "Smoking Status",
                ["no", "yes"]
            )

            region = st.selectbox(
                "Region",
                [
                    "northeast",
                    "northwest",
                    "southeast",
                    "southwest"
                ]
            )

        submitted = st.form_submit_button(
            "Predict Insurance Cost",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame(
            {
                "age": [age],
                "sex": [sex],
                "bmi": [bmi],
                "children": [children],
                "smoker": [smoker],
                "region": [region]
            }
        )

        # The saved best model is always available.
        st.subheader("Selected Model Prediction")

        prediction = best_model.predict(input_data)[0]

        st.success(
            f"Estimated insurance charges: ${prediction:,.2f}"
        )

        st.caption(
            f"Selected model: {best_model_name}"
        )

        st.divider()

        st.subheader("Compare Predictions Across All Models")

        # Load individual models if they have been saved.
        model_files = {
            "Linear Regression": "linear_regression.joblib",
            "Random Forest": "random_forest.joblib",
            "Gradient Boosting": "gradient_boosting.joblib"
        }

        available_models = {}

        for model_name, filename in model_files.items():

            model_path = MODEL_DIR / filename

            if model_path.exists():

                available_models[model_name] = joblib.load(
                    model_path
                )

        if available_models:

            prediction_rows = []

            for model_name, fitted_model in available_models.items():

                model_prediction = fitted_model.predict(
                    input_data
                )[0]

                prediction_rows.append(
                    {
                        "Model": model_name,
                        "Predicted Charges ($)": model_prediction
                    }
                )

            prediction_df = pd.DataFrame(prediction_rows)

            prediction_df["Predicted Charges ($)"] = (
                prediction_df["Predicted Charges ($)"]
                .map(lambda value: f"${value:,.2f}")
            )

            st.dataframe(
                prediction_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "Individual model files are not available yet. "
                "Run the updated training script to save all "
                "three models and enable side-by-side predictions."
            )

        st.caption(
            "These estimates are generated by machine-learning "
            "models trained on historical data. They are not "
            "actual insurance quotes."
        )


# =========================================================
# PAGE 5: ABOUT THE PROJECT
# =========================================================

elif page == "About the Project":

    st.header("About This Project")

    st.markdown(
        """
        ### Objective

        Build a machine-learning application that predicts
        medical insurance charges from demographic and
        lifestyle-related features.

        ### Dataset

        Medical Cost Personal Dataset.

        ### Input Features

        - Age
        - Sex
        - BMI
        - Number of children
        - Smoking status
        - Region

        ### Target Variable

        `charges` — medical insurance charges recorded in USD.

        ### Algorithms

        1. Linear Regression
        2. Random Forest Regressor
        3. Gradient Boosting Regressor

        ### Preprocessing

        - ColumnTransformer
        - OneHotEncoder
        - Pipeline

        ### Evaluation Metrics

        **MAE:** Average absolute difference between actual
        and predicted charges. Lower is better.

        **R²:** Measures the proportion of target variation
        explained by the model. Higher is generally better.

        ### Technology Stack

        - Python
        - pandas
        - scikit-learn
        - Matplotlib
        - Streamlit
        - Joblib

        ### Limitations

        The model learns patterns from a historical dataset.
        Its predictions may not generalize to every person,
        insurer, location, or current pricing policy.

        The application is intended for educational purposes
        and does not provide actual insurance quotes.
        """
    )

    st.divider()

    st.subheader("Current Best Model")

    st.write(f"**Model:** {best_model_name}")
    st.write(f"**Test MAE:** ${mae:,.2f}")
    st.write(f"**Test R²:** {r2:.4f}")

    st.info(
        "The model metrics describe performance on the held-out "
        "test dataset. They do not guarantee future accuracy."
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Insurance Cost Predictor | Built with Python, "
    "Scikit-learn and Streamlit"
)
