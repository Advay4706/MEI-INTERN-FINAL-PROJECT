import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def load_dataset():
    print("\nLoading dataset...\n")
    df = pd.read_csv("data/processed/master_dataset.csv")
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"],
        format="%Y%m%d %H:%M:%S"
    )
    start_time = df["DateTime"].min()
    df["ElapsedTime"] = (
        df["DateTime"] - start_time
    ).dt.total_seconds() / 60
    print("Dataset Loaded Successfully")
    print(f"\nRows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")
    return df


def prepare_features(df):
    print("\nPreparing Features...\n")
    features = [
        "RPM",
        "SpindleTemp",
        "XTemp",
        "YTemp",
        "ZTemp",
        "ElapsedTime"
    ]
    target = "Z_Error"
    X = df[features]
    y = df[target]
    print("Input Features")
    for feature in features:
        print("-", feature)
    print("\nTarget :", target)
    return X, y, features


def split_dataset(X, y):
    print("\nSplitting Dataset...\n")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
    print("Training Samples :", len(X_train))
    print("Testing Samples  :", len(X_test))
    return X_train, X_test, y_train, y_test


def train_linear_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    features
):
    print("\n" + "=" * 60)
    print("TRAINING LINEAR REGRESSION MODEL")
    print("=" * 60)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5,
        scoring="r2"
    )

    print("\nModel Performance\n")
    print(f"MAE                 : {mae:.6f}")
    print(f"RMSE                : {rmse:.6f}")
    print(f"R² Score            : {r2:.6f}")
    print(f"Average CV Score    : {cv_scores.mean():.6f}")

    print("\nIndividual CV Scores")
    for i, score in enumerate(cv_scores, start=1):
        print(f"Fold {i}: {score:.6f}")

    coefficients = pd.DataFrame(
        {
            "Feature": features,
            "Coefficient": model.coef_
        }
    )
    coefficients.to_csv(
        "data/processed/linear_regression_coefficients.csv",
        index=False
    )

    print("\nFeature Coefficients\n")
    print(coefficients)

    print("\nIntercept")
    print(model.intercept_)

    joblib.dump(model, "models/linear_regression.pkl")
    print("\nLinear Regression model saved successfully.")

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_scores": cv_scores,
        "coefficients": coefficients,
        "intercept": model.intercept_
    }


def train_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    features
):
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 10],
        "min_samples_split": [2, 5]
    }

    base_model = RandomForestRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=parameter_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1
    )

    print("\nFinding Best Hyperparameters...\n")
    grid_search.fit(X_train, y_train)

    model = grid_search.best_estimator_

    print("Best Parameters")
    print(grid_search.best_params_)

    best_params_df = pd.DataFrame([grid_search.best_params_])
    best_params_df.to_csv(
        "data/processed/best_random_forest_parameters.csv",
        index=False
    )

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    cv_scores = cross_val_score(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5,
        scoring="r2"
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_
        }
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False
    )

    feature_importance.to_csv(
        "data/processed/random_forest_feature_importance.csv",
        index=False
    )

    joblib.dump(model, "models/random_forest.pkl")

    print("\nRandom Forest Performance\n")
    print(f"MAE                 : {mae:.6f}")
    print(f"RMSE                : {rmse:.6f}")
    print(f"R² Score            : {r2:.6f}")
    print(f"Average CV Score    : {cv_scores.mean():.6f}")

    print("\nFeature Importance\n")
    print(feature_importance)

    print("\nRandom Forest model saved successfully.")

    return {
        "model": model,
        "predictions": predictions,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_scores": cv_scores,
        "feature_importance": feature_importance,
        "best_parameters": grid_search.best_params_
    }


def plot_actual_vs_predicted(
    actual,
    predicted,
    title,
    filename
):
    plt.figure(figsize=(8, 6))

    plt.scatter(actual, predicted, alpha=0.7)

    plt.plot(
        [actual.min(), actual.max()],
        [actual.min(), actual.max()],
        color="red",
        linewidth=2
    )

    plt.xlabel("Actual Z Error")
    plt.ylabel("Predicted Z Error")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"graphs/{filename}", dpi=300)
    plt.close()


def plot_feature_importance(feature_importance):
    plt.figure(figsize=(8, 6))

    plt.bar(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    plt.title("Random Forest Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=30)
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("graphs/12_feature_importance.png", dpi=300)
    plt.close()


def compare_models(lr_results, rf_results):
    comparison = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest"
        ],
        "MAE": [
            lr_results["mae"],
            rf_results["mae"]
        ],
        "RMSE": [
            lr_results["rmse"],
            rf_results["rmse"]
        ],
        "R2": [
            lr_results["r2"],
            rf_results["r2"]
        ]
    })

    comparison.to_csv(
        "data/processed/model_comparison.csv",
        index=False
    )

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(comparison)

    if rf_results["r2"] > lr_results["r2"]:
        print("\nBest Model : Random Forest")
    else:
        print("\nBest Model : Linear Regression")

    return comparison


def predict_new_reading(model):
    print("\n" + "=" * 60)
    print("PREDICTING THERMAL ERROR FOR NEW MACHINE DATA")
    print("=" * 60)

    new_data = pd.DataFrame({
        "RPM": [12000],
        "SpindleTemp": [52],
        "XTemp": [55],
        "YTemp": [55],
        "ZTemp": [66],
        "ElapsedTime": [420]
    })

    prediction = model.predict(new_data)

    print("\nInput Values")
    print(new_data)
    print(f"\nPredicted Z-Axis Error : {prediction[0]:.6f} mm")

    prediction_df = new_data.copy()
    prediction_df["Predicted_Z_Error"] = prediction
    prediction_df.to_csv(
        "data/processed/sample_prediction.csv",
        index=False
    )

    print("\nPrediction saved successfully.")


def main():
    os.makedirs("graphs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print("=" * 60)
    print("THERMAL ERROR COMPENSATION USING MACHINE LEARNING")
    print("=" * 60)

    df = load_dataset()

    X, y, features = prepare_features(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    lr_results = train_linear_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        features
    )

    rf_results = train_random_forest(
        X_train,
        X_test,
        y_train,
        y_test,
        features
    )

    plot_actual_vs_predicted(
        y_test,
        lr_results["predictions"],
        "Linear Regression : Actual vs Predicted",
        "08_actual_vs_predicted_lr.png"
    )

    plot_actual_vs_predicted(
        y_test,
        rf_results["predictions"],
        "Random Forest : Actual vs Predicted",
        "09_actual_vs_predicted_rf.png"
    )

    plot_feature_importance(
        rf_results["feature_importance"]
    )

    comparison = compare_models(lr_results, rf_results)

    predict_new_reading(rf_results["model"])

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nGenerated Files")

    print("\nGraphs")
    print("- Actual vs Predicted (Linear Regression)")
    print("- Actual vs Predicted (Random Forest)")
    print("- Feature Importance")

    print("\nModels")
    print("- linear_regression.pkl")
    print("- random_forest.pkl")

    print("\nCSV Files")
    print("- linear_regression_coefficients.csv")
    print("- random_forest_feature_importance.csv")
    print("- best_random_forest_parameters.csv")
    print("- model_comparison.csv")
    print("- sample_prediction.csv")

    print("\nThank you!")


if __name__ == "__main__":
    main()