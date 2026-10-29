# 🏥 Insurance Cost Prediction Using Machine Learning

A machine learning project that predicts individual medical insurance charges based on demographic information, lifestyle factors, and personal characteristics.

This project trains and compares three regression models — Linear Regression, Random Forest, and Gradient Boosting — and provides an interactive Streamlit dashboard for insurance cost prediction, model evaluation, and data visualization.

## 📌 Project Overview

Medical insurance charges can vary depending on several factors, including age, BMI, smoking habits, and other personal characteristics.

The objective of this project is to analyze historical insurance data, identify patterns using machine learning, and estimate insurance charges based on user-provided information.

### ✨ Key Features

* 🤖 Train and compare three machine learning regression models.
* 📊 Interactive dashboard built with Streamlit.
* 💰 Predict estimated insurance charges using personal characteristics.
* 📈 Compare model performance using MAE and R² score.
* 📉 Visualize relationships between age, smoking habits, and insurance charges.
* 🔍 Compare predictions from multiple trained models.
* ⚙️ Automated preprocessing of numerical and categorical features.
* 💾 Save and reuse trained models with Joblib.

## 🛠️ Tech Stack

| Technology   | Purpose                                 |
| ------------ | --------------------------------------- |
| Python       | Core programming language               |
| Pandas       | Data manipulation and analysis          |
| Scikit-learn | Data preprocessing and machine learning |
| Matplotlib   | Data visualization                      |
| Streamlit    | Interactive web application             |
| Joblib       | Saving and loading trained models       |
| Git & GitHub | Version control and project hosting     |

## 📂 Project Structure

```text
Insurance-Cost-Prediction-Using-Machine-Learning/
│
├── data/
│   └── insurance.csv
│
├── models/
│   ├── insurance_model.joblib
│   ├── linear_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   ├── model_comparison.csv
│   └── test_results.csv
│
├── app.py
├── train_model.py
├── check_data.py
├── requirements.txt
├── README.md
└── .gitignore
```

**File descriptions:**

* `data/insurance.csv` — Dataset used for training and evaluation.
* `models/insurance_model.joblib` — Best-performing trained model used for predictions.
* `models/linear_regression.joblib` — Trained Linear Regression pipeline.
* `models/random_forest.joblib` — Trained Random Forest pipeline.
* `models/gradient_boosting.joblib` — Trained Gradient Boosting pipeline.
* `models/model_comparison.csv` — Model performance comparison results.
* `models/test_results.csv` — Actual and predicted insurance charges for the test set.
* `train_model.py` — Preprocesses data, trains models, evaluates performance, and saves model artifacts.
* `app.py` — Runs the interactive Streamlit dashboard.
* `check_data.py` — Inspects the dataset.
* `requirements.txt` — Lists the project's Python dependencies.

*The individual model filenames assume all three trained pipelines have been saved by the training script.*

## 📊 Dataset

This project uses the **Medical Cost Personal Dataset**, which contains demographic and lifestyle information alongside individual medical insurance charges.

### Dataset Features

| Feature    | Description                          |
| ---------- | ------------------------------------ |
| `age`      | Age of the individual                |
| `sex`      | Sex category recorded in the dataset |
| `bmi`      | Body Mass Index                      |
| `children` | Number of dependent children         |
| `smoker`   | Smoking status                       |
| `region`   | Residential region                   |
| `charges`  | Medical insurance charges            |

**Target variable:** `charges`

The six remaining columns are used as input features to predict insurance charges.

### Data Preprocessing

The machine learning pipeline performs the following operations:

1. Separates input features from the target variable.
2. Encodes categorical features using One-Hot Encoding.
3. Preserves numerical features for model training.
4. Splits the dataset into training and testing sets.
5. Trains and evaluates multiple regression models.

The categorical encoder uses `OneHotEncoder(handle_unknown="ignore")` to handle previously unseen categories during prediction.

## 🧠 Machine Learning Models

Three regression algorithms are trained and compared.

| Model                       | Description                                                                                      |
| --------------------------- | ------------------------------------------------------------------------------------------------ |
| Linear Regression           | Establishes a baseline by modelling linear relationships between features and insurance charges. |
| Random Forest Regressor     | Combines predictions from multiple decision trees to model complex relationships.                |
| Gradient Boosting Regressor | Builds an ensemble of decision trees sequentially to improve predictive performance.             |

### Evaluation Metrics

**1. Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted insurance charges. A lower MAE indicates more accurate predictions on average.

**2. R² Score**

Measures how well the model explains the variation in insurance charges. A score closer to 1 generally indicates a better fit.

### 📈 Model Performance

The following results were obtained during the initial evaluation on a held-out test set.

| Model             |       MAE | R² Score |
| ----------------- | --------: | -------: |
| Linear Regression | $4,181.19 |   0.7836 |
| Random Forest     | $2,546.50 |   0.8637 |
| Gradient Boosting | $2,406.17 |   0.8793 |

**Best-performing model: Gradient Boosting Regressor**

Gradient Boosting achieved the lowest MAE and highest R² score among the three models in this evaluation. It was selected as the best-performing model for the prediction dashboard.

*Note: These are the initial evaluation results. Results may change if the dataset, preprocessing, model parameters, or evaluation procedure changes.*

## 📊 Dashboard Features

The Streamlit application provides five main sections.

### 1. Dashboard Overview

Displays dataset information, model performance metrics, and a preview of the dataset.

### 2. Model Comparison

Compares Linear Regression, Random Forest, and Gradient Boosting using performance tables and charts.

### 3. Data Visualizations

Includes visualizations such as:

* Age vs. Insurance Charges
* Smoking Status vs. Average Insurance Charges
* Actual vs. Predicted Insurance Charges

### 4. Insurance Prediction

Allows users to enter the following details to estimate insurance charges:

* Age
* Sex
* BMI
* Number of children
* Smoking status
* Region

The application processes these inputs using the trained machine learning pipeline and displays an estimated insurance charge.

### 5. About the Project

Explains the project's objectives, technologies, machine learning models, and limitations.

When all three individual model files are available, the prediction interface can also compare estimates from the different models for the same input.

## 🚀 Installation and Setup

Follow these steps to run the project locally.

### Prerequisites

* Python 3.10 or another version compatible with the installed dependencies
* Git
* Visual Studio Code or another Python-compatible editor

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Insurance-Cost-Prediction-Using-Machine-Learning.git
```

Navigate to the project directory:

```bash
cd Insurance-Cost-Prediction-Using-Machine-Learning
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Virtual Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Train the Models

If the trained model files are not already available, run:

```bash
python train_model.py
```

This executes the training pipeline and saves the model artifacts and evaluation results configured in the script.

### Step 6: Launch the Dashboard

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open this URL in your browser to access the dashboard.

## 💻 Example Prediction

Consider an individual with the following characteristics:

| Feature  | Example Value |
| -------- | ------------- |
| Age      | 30            |
| Sex      | Female        |
| BMI      | 25.5          |
| Children | 1             |
| Smoker   | No            |
| Region   | Southeast     |

The application uses these inputs to generate an estimated insurance charge.

The predicted amount depends on the trained model and should not be interpreted as an official insurance quotation.

## 🔮 Future Improvements

* [ ] Add RMSE (Root Mean Squared Error) to model evaluation.
* [ ] Implement 5-fold cross-validation for more robust model comparison.
* [ ] Perform hyperparameter tuning to improve model performance.
* [ ] Add feature importance analysis.
* [ ] Provide more detailed explanations of individual predictions.
* [ ] Improve dashboard design and interactivity.
* [ ] Deploy the Streamlit application online.
* [ ] Add additional evaluation charts and model diagnostics.

## ⚠️ Limitations

* Predictions are estimates based on patterns in the available dataset.
* Model accuracy depends on the quality and representativeness of the training data.
* Historical relationships do not establish causation.
* The dataset may not reflect current insurance pricing, policies, or regional differences.
* Predictions should not be treated as guaranteed insurance quotes or professional financial advice.

## 🎯 Learning Outcomes

Through this project, I explored:

* Data inspection and preprocessing using Pandas and Scikit-learn.
* Encoding categorical variables for machine learning.
* Training and comparing regression algorithms.
* Evaluating models using MAE and R² score.
* Saving and loading trained pipelines using Joblib.
* Building an interactive machine learning dashboard with Streamlit.
* Visualizing datasets and prediction results.
* Organizing a Python project for GitHub.

## 👨‍💻 Author

**Shravan More**

## 📄 License

This project was developed for educational purposes and machine learning practice. Before redistributing the project or dataset, verify the applicable dataset licensing terms. A separate project license can be added if required.
