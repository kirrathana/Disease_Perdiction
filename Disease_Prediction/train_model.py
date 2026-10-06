"""
Heart Disease Prediction - Model Training Script

This script loads the heart disease dataset, performs preprocessing,
trains multiple classification models, evaluates them, and saves the
best model along with evaluation artifacts.
"""

import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "heart.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TARGET_COLUMN = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Plot styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11


def ensure_models_dir():
    """Create models directory if it does not exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_analyze_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset and display exploratory summary statistics.

    Args:
        filepath: Path to the CSV dataset file.

    Returns:
        Loaded pandas DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. "
            "Please place heart.csv in the dataset/ folder."
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load dataset: {exc}")

    print("=" * 60)
    print("STEP 1: DATASET ANALYSIS")
    print("=" * 60)

    print("\n--- First 5 Rows ---")
    print(df.head())

    print(f"\n--- Shape ---\n{df.shape[0]} rows x {df.shape[1]} columns")

    print("\n--- Columns ---")
    print(list(df.columns))

    print("\n--- Data Types ---")
    print(df.dtypes)

    print("\n--- Descriptive Statistics ---")
    print(df.describe())

    print("\n--- Null Values ---")
    null_counts = df.isnull().sum()
    print(null_counts if null_counts.sum() > 0 else "No null values found.")

    print(f"\n--- Duplicate Rows ---\n{df.duplicated().sum()}")

    print("\n--- Target Class Distribution ---")
    target_dist = df[TARGET_COLUMN].value_counts()
    print(target_dist)
    print(f"\nClass proportions:\n{df[TARGET_COLUMN].value_counts(normalize=True)}")

    return df


def preprocess_data(df: pd.DataFrame):
    """
    Preprocess data: remove duplicates, handle missing values,
    split features/target, and scale features.

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names).
    """
    print("\n" + "=" * 60)
    print("STEP 3: PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # Remove duplicates
    duplicates_before = df.duplicated().sum()
    df.drop_duplicates(inplace=True)
    print(f"Removed {duplicates_before} duplicate row(s).")

    # Handle missing values
    missing_before = df.isnull().sum().sum()
    if missing_before > 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        print(f"Filled {missing_before} missing value(s) with column medians.")
    else:
        print("No missing values to handle.")

    # Separate features and target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    feature_names = list(X.columns)

    print(f"Features: {feature_names}")
    print(f"Target distribution after preprocessing:\n{y.value_counts()}")

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def get_models():
    """
    Return dictionary of model name -> sklearn/xgboost classifier.

    Returns:
        Dict mapping model names to untrained estimators.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            use_label_encoder=False,
        ),
    }


def evaluate_model(y_true, y_pred, y_prob):
    """
    Compute classification metrics for a single model.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities for positive class.

    Returns:
        Dict of metric name -> score.
    """
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train all models and return comparison results and trained estimators.

    Returns:
        Tuple of (results_df, trained_models dict, best model name).
    """
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING MULTIPLE MODELS")
    print("=" * 60)

    models = get_models()
    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model(y_test, y_pred, y_prob)
            metrics["Model"] = name
            results.append(metrics)
            trained_models[name] = model
            print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
        except Exception as exc:
            print(f"  Error training {name}: {exc}")

    results_df = pd.DataFrame(results).set_index("Model")
    print("\n--- Model Comparison Table ---")
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

    best_name = results_df["ROC-AUC"].idxmax()
    return results_df, trained_models, best_name


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Disease", "Heart Disease"],
        yticklabels=["No Disease", "Heart Disease"],
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Generate and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#2E86AB", lw=2, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, save_path):
    """Generate and save precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#A23B72", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, feature_names, save_path):
    """Generate and save Random Forest feature importance plot."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    colors = sns.color_palette("viridis", len(feature_names))
    plt.bar(
        range(len(importances)),
        importances[indices],
        color=colors,
        edgecolor="white",
    )
    plt.xticks(
        range(len(importances)),
        [feature_names[i] for i in indices],
        rotation=45,
        ha="right",
    )
    plt.title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("\n--- Top Important Features (Random Forest) ---")
    for rank, idx in enumerate(indices[:5], start=1):
        print(f"  {rank}. {feature_names[idx]}: {importances[idx]:.4f}")


def generate_eda_plots(df, feature_names):
    """Generate EDA plots required for the models/ folder."""
    print("\nGenerating EDA plots...")

    # Target distribution
    plt.figure(figsize=(8, 5))
    counts = df[TARGET_COLUMN].value_counts()
    colors = ["#2ECC71", "#E74C3C"]
    bars = plt.bar(
        ["No Heart Disease (0)", "Heart Disease (1)"],
        counts.values,
        color=colors,
        edgecolor="white",
    )
    for bar, val in zip(bars, counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(val), ha="center", fontweight="bold",
        )
    plt.title("Target Distribution", fontsize=14, fontweight="bold")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODELS_DIR, "target_distribution.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5,
    )
    plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODELS_DIR, "correlation_heatmap.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    # Feature histograms
    feature_cols = [c for c in feature_names if c in df.columns]
    n_cols = 3
    n_rows = (len(feature_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="#3498DB", edgecolor="white")
        axes[i].set_title(col, fontweight="bold")
    for j in range(len(feature_cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Feature Histograms", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(
        os.path.join(MODELS_DIR, "feature_histograms.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()

    print("EDA plots saved to models/")


def main():
    """Main training pipeline."""
    ensure_models_dir()

    # Step 1 & 2: Load and analyze
    df = load_and_analyze_data(DATASET_PATH)
    feature_names = [c for c in df.columns if c != TARGET_COLUMN]

    # Generate EDA plots
    generate_eda_plots(df, feature_names)

    # Step 3: Preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

    # Step 4: Train models
    results_df, trained_models, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )

    best_model = trained_models[best_name]
    y_pred_best = best_model.predict(X_test)
    y_prob_best = best_model.predict_proba(X_test)[:, 1]
    best_metrics = evaluate_model(y_test, y_pred_best, y_prob_best)

    # Step 5: Model evaluation plots
    print("\n" + "=" * 60)
    print("STEP 5: MODEL EVALUATION")
    print("=" * 60)

    print("\n--- Classification Report ---")
    print(classification_report(
        y_test, y_pred_best,
        target_names=["No Disease", "Heart Disease"],
    ))

    plot_confusion_matrix(
        y_test, y_pred_best,
        os.path.join(MODELS_DIR, "confusion_matrix.png"),
    )
    plot_roc_curve(
        y_test, y_prob_best,
        os.path.join(MODELS_DIR, "roc_curve.png"),
    )
    plot_precision_recall_curve(
        y_test, y_prob_best,
        os.path.join(MODELS_DIR, "precision_recall_curve.png"),
    )
    print("Evaluation plots saved to models/")

    # Step 6: Feature importance (Random Forest)
    print("\n" + "=" * 60)
    print("STEP 6: FEATURE IMPORTANCE")
    print("=" * 60)

    if "Random Forest" in trained_models:
        plot_feature_importance(
            trained_models["Random Forest"],
            feature_names,
            os.path.join(MODELS_DIR, "feature_importance.png"),
        )

    # Step 7: Save best model
    print("\n" + "=" * 60)
    print("STEP 7: BEST MODEL SELECTION")
    print("=" * 60)

    print(f"\nBest Model: {best_name}")
    print(f"Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"Precision: {best_metrics['Precision']:.4f}")
    print(f"Recall:    {best_metrics['Recall']:.4f}")
    print(f"F1 Score:  {best_metrics['F1 Score']:.4f}")
    print(f"ROC-AUC:   {best_metrics['ROC-AUC']:.4f}")

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\nBest model saved to: {model_path}")

    # Save metadata for Streamlit app
    metadata = {
        "best_model_name": best_name,
        "feature_names": feature_names,
        "metrics": best_metrics,
        "results_df": results_df.to_dict(),
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "model_metadata.pkl"))
    print("Model metadata saved.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
