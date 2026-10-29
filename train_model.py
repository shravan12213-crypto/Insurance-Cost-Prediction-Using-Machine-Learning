import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset

df = pd.read_csv("data/insurance.csv")

print("Dataset loaded:", df.shape)


# 2. Separate features and target

X = df.drop("charges", axis=1)
y = df["charges"]


# 3. Define feature columns

numerical_features = ["age", "bmi", "children"]

categorical_features = ["sex", "smoker", "region"]


# 4. Preprocessing

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ],
        remainder="passthrough"
    )


# 5. Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 6. Define models

models = {

    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )

}


# 7. Train and evaluate models

results = []
trained_models = {}

for name, algorithm in models.items():

    print(f"\nTraining {name}...")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("regressor", algorithm)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "R2 Score": r2
    })

    trained_models[name] = pipeline
    
    # Save each trained model for dashboard comparisons

    model_filenames = {
        "Linear Regression": "linear_regression.joblib",
        "Random Forest": "random_forest.joblib",
        "Gradient Boosting": "gradient_boosting.joblib"
    }

    joblib.dump(
        pipeline,
        os.path.join("models", model_filenames[name])
    )

    print(f"MAE: ${mae:,.2f}")
    print(f"R2 Score: {r2:.4f}")


# 8. Compare model results

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\nMODEL COMPARISON")
print("=" * 55)
print(results_df.to_string(index=False))


# 9. Select the best model using MAE

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

print(f"\nBest model based on test MAE: {best_model_name}")


# 10. Save the best model

os.makedirs("models", exist_ok=True)

joblib.dump(
    best_model,
    "models/insurance_model.joblib"
)


# 11. Save comparison results

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)


# 12. Save best model's test predictions

best_predictions = best_model.predict(X_test)

test_results = X_test.copy()

test_results["actual_charges"] = y_test
test_results["predicted_charges"] = best_predictions

test_results.to_csv(
    "models/test_results.csv",
    index=False
)


print("\nBest model saved successfully!")
print("Model: models/insurance_model.joblib")
print("Comparison: models/model_comparison.csv")
print("Predictions: models/test_results.csv")