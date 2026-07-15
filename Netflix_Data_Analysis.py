"""
House Price Prediction - Dashboard Version
--------------------------------------------
Trains a Linear Regression model and renders a single-image dashboard
summarizing dataset stats, model performance, and diagnostic plots.

Usage:
    python house_price_dashboard.py
    python house_price_dashboard.py --input house_price.csv --target price
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    parser = argparse.ArgumentParser(description="Train model and build a results dashboard.")
    parser.add_argument("--input", default="house_price.csv", help="Path to input CSV file")
    parser.add_argument("--target", default="price", help="Name of the target column")
    parser.add_argument("--output-dir", default="output", help="Directory to save model/dashboard")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data for testing")
    return parser.parse_args()


def load_data(filepath: str, target: str) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        sys.exit(f"Error: file not found -> {filepath}")
    df = pd.read_csv(path)
    if target not in df.columns:
        sys.exit(f"Error: target column '{target}' not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[target])
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]), numeric_features, categorical_features


def get_feature_names(pipeline, numeric_features, categorical_features):
    """Recover readable feature names after one-hot encoding."""
    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(categorical_features)) if categorical_features else []
    return numeric_features + cat_names


def build_dashboard(df, y_test, y_pred, metrics, cv_scores, coef_series, output_dir: Path, target: str):
    """Render a single dashboard image summarizing data, performance, and diagnostics."""
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("House Price Prediction — Model Dashboard", fontsize=18, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # 1. Actual vs Predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, y_pred, alpha=0.6, edgecolor="k", color="steelblue")
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax1.plot(lims, lims, "r--", label="Perfect fit")
    ax1.set_xlabel("Actual Price")
    ax1.set_ylabel("Predicted Price")
    ax1.set_title("Actual vs Predicted")
    ax1.legend()

    # 2. Residuals distribution
    residuals = y_test.values - y_pred
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=20, color="darkorange", edgecolor="black")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_title("Residuals Distribution")
    ax2.set_xlabel("Actual - Predicted")
    ax2.set_ylabel("Frequency")

    # 3. Target distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df[target], bins=20, color="seagreen", edgecolor="black")
    ax3.set_title(f"{target.title()} Distribution")
    ax3.set_xlabel(target.title())
    ax3.set_ylabel("Count")

    # 4. Feature coefficients
    ax4 = fig.add_subplot(gs[1, 0:2])
    top_coef = coef_series.reindex(coef_series.abs().sort_values(ascending=False).index).head(10)
    colors = ["crimson" if v < 0 else "steelblue" for v in top_coef.values]
    ax4.barh(top_coef.index[::-1], top_coef.values[::-1], color=colors[::-1])
    ax4.set_title("Top Feature Impact on Price (Standardized Coefficients)")
    ax4.set_xlabel("Coefficient Value")

    # 5. Metrics summary panel
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_text = (
        f"MODEL PERFORMANCE\n"
        f"{'-'*28}\n"
        f"MAE:   {metrics['mae']:,.0f}\n"
        f"RMSE:  {metrics['rmse']:,.0f}\n"
        f"R2 (test):   {metrics['r2']:.3f}\n"
        f"R2 (5-fold CV): {cv_scores.mean():.3f} \u00b1 {cv_scores.std():.3f}\n\n"
        f"DATASET\n"
        f"{'-'*28}\n"
        f"Rows: {len(df)}\n"
        f"Avg {target}: {df[target].mean():,.0f}\n"
        f"Min / Max: {df[target].min():,.0f} / {df[target].max():,.0f}"
    )
    ax5.text(0.02, 0.98, summary_text, va="top", ha="left", fontsize=11,
              family="monospace", bbox=dict(boxstyle="round", facecolor="whitesmoke", edgecolor="gray"))

    output_dir.mkdir(parents=True, exist_ok=True)
    dashboard_path = output_dir / "dashboard.png"
    fig.savefig(dashboard_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return dashboard_path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    df = load_data(args.input, args.target)
    print("Dataset preview:")
    print(df.head())

    X = df.drop(columns=[args.target])
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )

    pipeline, numeric_features, categorical_features = build_pipeline(X)

    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="r2")
    print(f"\nCross-validated R2 (train, 5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}
    print(f"MAE: {mae:,.2f} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")

    feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    coefs = pipeline.named_steps["model"].coef_
    coef_series = pd.Series(coefs, index=feature_names)

    dashboard_path = build_dashboard(df, y_test, y_pred, metrics, cv_scores, coef_series, output_dir, args.target)
    print(f"\nDashboard saved to: {dashboard_path}")

    model_path = output_dir / "house_price_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
