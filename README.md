# 🏥 Insurance Cost Prediction Using Machine Learning

A machine learning project that predicts medical insurance costs based on personal attributes such as age, BMI, smoking status, gender, number of children, and residential region.

The project uses **Python and Scikit-learn** to train and evaluate regression models and **Streamlit** to provide an interactive web application for insurance cost prediction and data visualization.

## ✨ Features

* 📊 **Data Analysis:** Explore the medical insurance dataset using visualizations.
* ⚙️ **Data Preprocessing:** Automatically handle categorical and numerical features using Scikit-learn.
* 🤖 **Machine Learning:** Train and compare Linear Regression and Random Forest Regression models.
* 📈 **Model Evaluation:** Evaluate model performance using Mean Absolute Error (MAE) and R² score.
* 💰 **Cost Prediction:** Estimate insurance charges using user-provided information.
* 📉 **Data Visualization:** Visualize relationships between age, smoking status, and insurance charges.
* 🖥️ **Interactive Dashboard:** Use a simple, user-friendly Streamlit interface.

## 🛠️ Technologies Used

| Technology   | Purpose                                            |
| ------------ | -------------------------------------------------- |
| Python       | Core programming language                          |
| Pandas       | Data manipulation and analysis                     |
| NumPy        | Numerical operations                               |
| Scikit-learn | Data preprocessing, model training, and evaluation |
| Matplotlib   | Data visualization                                 |
| Joblib       | Saving and loading trained models                  |
| Streamlit    | Interactive web application                        |

## 📂 Project Structure

```text
insurance-cost-predictor-ml/
│
├── insurance.csv          # Medical insurance dataset
├── train_model.py         # Data preprocessing and model training
├── app.py                 # Streamlit web application
├── insurance_model.pkl    # Saved trained model pipeline
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

*The files `insurance_model.pkl` and other generated files will be created during development. The exact structure may change as the project evolves.*

## 📊 Dataset

This project uses the **Medical Cost Personal Dataset** available on Kaggle.

**Dataset:** [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

The dataset contains information about individuals and their medical insurance charges.

### Dataset Features

| Feature    | Description                                 |
| ---------- | ------------------------------------------- |
| `age`      | Age of the individual                       |
| `sex`      | Gender of the individual                    |
| `bmi`      | Body Mass Index                             |
| `children` | Number of dependent children                |
| `smoker`   | Smoking status                              |
| `region`   | Residential region                          |
| `charges`  | Medical insurance charges (target variable) |

The `charges` column is the target the machine learning models learn to predict.

## 🧠 Machine Learning Models

The project uses regression algorithms to estimate medical insurance charges.

### 1. Linear Regression

Linear Regression serves as the baseline model. It learns a linear relationship between the input features and insurance charges.

### 2. Random Forest Regression

Random Forest Regression combines multiple decision trees to model potentially nonlinear relationships between input features and insurance charges.

Both models will be evaluated on the same test dataset to compare their predictive performance.

### Data Preprocessing

The project uses Scikit-learn's preprocessing tools:

* `ColumnTransformer` to apply different transformations to numerical and categorical features.
* `OneHotEncoder` to convert categorical features into numerical representations.
* `Pipeline` to combine preprocessing and model training into a consistent workflow.

This ensures that the same preprocessing steps are applied during both training and prediction.

## 📈 Model Evaluation

The models are evaluated using the following metrics:

| Metric                    | Description                                                                                                                        |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Mean Absolute Error (MAE) | Measures the average absolute difference between actual and predicted insurance charges. Lower values indicate better performance. |
| R² Score                  | Measures how well the model explains variation in insurance charges. A score closer to 1 generally indicates a better fit.         |

The final model will be selected based on its evaluation results.

*Actual evaluation metrics will be added after training and testing the models.*

## 🚀 Getting Started

Follow these steps to run the project locally.

### Prerequisites

* Python 3.10 or a compatible version
* Visual Studio Code or another Python editor
* Git (optional, for version control)

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/insurance-cost-predictor-ml.git
```

Replace `YOUR-USERNAME` with your GitHub username.

Navigate to the project directory:

```bash
cd insurance-cost-predictor-ml
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available yet, install the libraries directly:

```bash
pip install pandas numpy scikit-learn matplotlib joblib streamlit
```

### 4. Add the Dataset

Download `insurance.csv` from Kaggle and place it in the project directory.

### 5. Train the Models

Run the training script:

```bash
python train_model.py
```

The script will preprocess the data, train and evaluate the regression models, and save the selected trained model.

### 6. Launch the Web Application

After the trained model has been generated, start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser, allowing you to enter personal attributes and obtain an estimated insurance cost.

## 📉 Visualizations

The dashboard is planned to include:

* **Age vs. Insurance Charges:** Explore how insurance costs vary with age.
* **Smoking Status vs. Average Charges:** Compare average insurance costs between smokers and non-smokers.
* **Actual vs. Predicted Charges:** Assess the model's predictions against actual charges in the test dataset.
* **Model Performance Comparison:** Compare MAE and R² scores for the trained regression models.

The visualizations will be generated from the dataset and actual model evaluation results.

## 🎯 Project Objectives

* Understand the end-to-end machine learning workflow.
* Apply data preprocessing techniques to real-world tabular data.
* Train and compare regression algorithms.
* Evaluate model performance using standard regression metrics.
* Build an interactive machine learning application.
* Present predictions and insights through data visualizations.

## 🔮 Future Improvements

* Experiment with additional regression algorithms, such as Gradient Boosting.
* Improve prediction performance through hyperparameter tuning.
* Add feature importance analysis.
* Enhance the dashboard with additional data visualizations.
* Deploy the application online for public access.

## ⚠️ Disclaimer

This project is intended for educational purposes only. Predictions are estimates based on patterns in the dataset and should not be treated as actual insurance quotations or professional financial advice.

## 👨‍💻 Author

**Shravan More**

Developed as a machine learning project using Python, Scikit-learn, and Streamlit.

---

⭐ If you find this project useful, consider giving the repository a star!
