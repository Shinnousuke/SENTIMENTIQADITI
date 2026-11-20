
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from transformers import pipeline, AutoConfig

# =====================================================
# 1. PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

TEST_DIR = BASE_DIR / "data" / "test"

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "indic_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

LABELS = ["Negative", "Neutral", "Positive"]


# =====================================================
# 2. READ JSON AND JSONL FILES
# =====================================================

def read_json_file(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

    if not content:
        return []

    try:
        data = json.loads(content)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    records = []

    for line_number, line in enumerate(content.splitlines(), start=1):

        if not line.strip():
            continue

        try:
            records.append(json.loads(line))

        except json.JSONDecodeError as error:
            print(f"Invalid line {line_number} in {file_path.name}: {error}")

    return records


# =====================================================
# 3. LOAD ALL LANGUAGE DATASETS
# =====================================================

def load_dataset():

    if not TEST_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {TEST_DIR}")

    files = sorted(TEST_DIR.glob("*.json"))

    if not files:
        raise FileNotFoundError("No JSON files found inside data/test.")

    all_records = []

    for file_path in files:

        language = file_path.stem.lower()

        print(f"Loading {file_path.name}...")

        records = read_json_file(file_path)

        for record in records:

            if not isinstance(record, dict):
                continue

            review = record.get("INDIC REVIEW")
            actual = record.get("LABEL")

            if review is None or actual is None:
                continue

            review = str(review).strip()
            actual = str(actual).strip().capitalize()

            if review and actual in LABELS:

                all_records.append({
                    "language": language,
                    "review": review,
                    "actual_sentiment": actual
                })

    if not all_records:
        raise ValueError(
            "No valid reviews found. Check the INDIC REVIEW and LABEL columns."
        )

    df = pd.DataFrame(all_records)

    print("\nDATASET LOADED")
    print("Total reviews:", len(df))
    print("\nLanguage distribution:")
    print(df["language"].value_counts())

    return df


# =====================================================
# 4. LOAD MULTILINGUAL MODEL
# =====================================================


from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

def load_model():

    MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    print("\nLoading multilingual sentiment model...")

    # The sentencepiece library is blocked by this machine's Application
    # Control policy, so use the fast (Rust) tokenizer from the base XLM-R
    # model, which shares this model's vocabulary.
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    sentiment_model = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1
    )

    print("Model loaded successfully!")

    return sentiment_model, model.config


# =====================================================
# 5. MAP MODEL LABELS
# =====================================================

def normalize_label(label, config):

    label = str(label).strip()

    if label.upper().startswith("LABEL_"):

        label_id = int(label.split("_")[-1])
        label = str(config.id2label[label_id])

    label = label.lower()

    if "negative" in label:
        return "Negative"

    if "neutral" in label:
        return "Neutral"

    if "positive" in label:
        return "Positive"

    raise ValueError(f"Unknown model label: {label}")


# =====================================================
# 6. PREDICT SENTIMENT
# =====================================================

def predict_sentiment(df, model, config):

    reviews = df["review"].tolist()

    predictions = []

    batch_size = 16

    print("\nStarting predictions...")

    for start in range(0, len(reviews), batch_size):

        batch = reviews[start:start + batch_size]

        outputs = model(batch, batch_size=batch_size)

        for output in outputs:

            predictions.append(
                normalize_label(output["label"], config)
            )

        print(
            f"Processed {min(start + batch_size, len(reviews))}"
            f"/{len(reviews)}"
        )

    df["predicted_sentiment"] = predictions

    return df


# =====================================================
# 7. OVERALL EVALUATION
# =====================================================

def evaluate_overall(df):

    actual = df["actual_sentiment"]
    predicted = df["predicted_sentiment"]

    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 50)
    print("OVERALL SENTIMENT EVALUATION")
    print("=" * 50)

    print("Total reviews:", len(df))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=LABELS,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=LABELS
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=LABELS,
        columns=LABELS
    )

    print("\nConfusion Matrix:")
    print(matrix_df)

    return accuracy, macro_f1, matrix_df


# =====================================================
# 8. LANGUAGE-WISE EVALUATION
# =====================================================

def evaluate_languages(df):

    results = []

    for language, group in df.groupby("language"):

        accuracy = accuracy_score(
            group["actual_sentiment"],
            group["predicted_sentiment"]
        )

        macro_f1 = f1_score(
            group["actual_sentiment"],
            group["predicted_sentiment"],
            labels=LABELS,
            average="macro",
            zero_division=0
        )

        results.append({
            "language": language,
            "total_reviews": len(group),
            "accuracy": accuracy,
            "macro_f1": macro_f1
        })

    results_df = pd.DataFrame(results)

    print("\nLANGUAGE-WISE PERFORMANCE")
    print(results_df.to_string(index=False))

    return results_df


# =====================================================
# 9. SAVE RESULTS
# =====================================================

def save_results(df, language_results, matrix_df, accuracy, macro_f1):

    from sklearn.metrics import precision_recall_fscore_support

    # File names below match what backend/main.py expects.

    df = df.copy()
    df["correct"] = df["actual_sentiment"] == df["predicted_sentiment"]

    df.drop(columns=["correct"]).to_csv(
        OUTPUT_DIR / "all_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    df[~df["correct"]].drop(columns=["correct"]).to_csv(
        OUTPUT_DIR / "incorrect_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Language-wise metrics (with precision and recall added)
    rows = []
    for language, group in df.groupby("language"):
        p, r, f, _ = precision_recall_fscore_support(
            group["actual_sentiment"],
            group["predicted_sentiment"],
            labels=LABELS,
            average="macro",
            zero_division=0
        )
        rows.append({
            "language": language,
            "total_reviews": len(group),
            "accuracy": round(float(accuracy_score(
                group["actual_sentiment"], group["predicted_sentiment"])), 4),
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4)
        })

    lang_df = pd.DataFrame(rows)
    lang_df.to_csv(
        OUTPUT_DIR / "language_wise_metrics.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Overall metrics
    p, r, f, _ = precision_recall_fscore_support(
        df["actual_sentiment"],
        df["predicted_sentiment"],
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    metrics = {
        "model": MODEL_NAME,
        "total_reviews": int(len(df)),
        "incorrect_predictions": int((~df["correct"]).sum()),
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(macro_f1), 4)
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig"
    )

    # Optional PNG charts (need: pip install matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        chart_dir = OUTPUT_DIR / "charts"
        chart_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix_df.values, cmap="Blues")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS)
        ax.set_yticklabels(LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, int(matrix_df.values[i, j]),
                        ha="center", va="center")
        fig.tight_layout()
        fig.savefig(chart_dir / "confusion_matrix.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(lang_df["language"], lang_df["accuracy"])
        ax.set_ylim(0, 1)
        ax.set_title("Accuracy by language")
        fig.tight_layout()
        fig.savefig(chart_dir / "language_accuracy.png", dpi=120)
        plt.close(fig)

    except ImportError:
        print("matplotlib not installed - skipping PNG charts.")

    print("\nAll evaluation files saved in:")
    print(OUTPUT_DIR)


# =====================================================
# 10. MAIN
# =====================================================

def main():

    print("SENTIMENTIQ MULTILINGUAL EVALUATION")
    print("=" * 50)

    df = load_dataset()

    model, config = load_model()

    df = predict_sentiment(df, model, config)

    accuracy, macro_f1, matrix_df = evaluate_overall(df)

    language_results = evaluate_languages(df)

    save_results(
        df,
        language_results,
        matrix_df,
        accuracy,
        macro_f1
    )

    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
