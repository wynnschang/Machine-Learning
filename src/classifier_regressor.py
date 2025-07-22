import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import os

def prepare_train_set(past_df, current_df):
    labels = current_df[["SK_ID_CURR", "BALANCE_CLASS"]]
    train_df = past_df.merge(labels, on="SK_ID_CURR", how="inner")

    train_df.drop(columns=["AVG_BALANCE"], inplace=True, errors="ignore")
    if "BALANCE_CLASS_x" in train_df.columns and "BALANCE_CLASS_y" in train_df.columns:
        train_df.drop(columns=["BALANCE_CLASS_x"], inplace=True)
        train_df.rename(columns={"BALANCE_CLASS_y": "BALANCE_CLASS"}, inplace=True)

    X = train_df.drop(columns=["SK_ID_CURR", "BALANCE_CLASS"])
    y = train_df["BALANCE_CLASS"]
    return X, y

def train_models(X_train, y_train, X_val, y_val):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, multi_class="multinomial"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42),
        "LightGBM": LGBMClassifier(random_state=42)
    }

    results = []
    best_model = None
    best_model_name = ""
    best_score = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")

        print(f"\n=== {name} ===")
        print(classification_report(y_val, y_pred))

        results.append({"Model": name, "Accuracy": acc, "Macro F1-score": f1})
        if f1 > best_score:
            best_model = model
            best_model_name = name
            best_score = f1

    results_df = pd.DataFrame(results).sort_values(by=["Macro F1-score", "Accuracy"], ascending=False)
    sns.barplot(data=results_df, x="Macro F1-score", y="Model")
    plt.title("Model Ranking by Macro F1 (Validation)")
    plt.tight_layout()
    plt.show()

    return best_model, best_model_name, results_df

def evaluate_model(model, X, y_true, label="", precomputed=False):
    y_pred = X if precomputed else model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\nPerformance on {label} Data:")
    print(classification_report(y_true, y_pred))
    print(f"Accuracy: {acc:.4f}, Macro F1-score: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['0', '1', '2'], yticklabels=['0', '1', '2'])
    plt.title(f"Confusion Matrix - {label}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    return y_pred

def plot_feature_importance(model, X_train, model_name):

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        print(f"\nTop 10 Features - {model_name}:\n")
        print(importance_df.head(10))

        os.makedirs("../results", exist_ok=True)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=importance_df.head(10), x="Importance", y="Feature", palette="viridis")
        plt.title(f"Top 10 Features - {model_name}")
        plt.tight_layout()
        plt.savefig("../results/classifier_feature_importance.png")
        plt.show()
    else:
        print(f"{model_name} does not support feature_importances_.")