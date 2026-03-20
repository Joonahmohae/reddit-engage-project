from transformers import pipeline, logging as hf_logging
from huggingface_hub import logging as hub_logging
from .features import make_all_feature_tables
from .clean import preview_df
import pandas as pd
import os


hf_logging.set_verbosity_error()
hub_logging.set_verbosity_error()


OUTPUT_DIR = "outputs"


def load_sentiment_model():
    print("Loading sentiment model")
    clf = pipeline(
        "sentiment-analysis",
        model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    print("Sentiment model loaded.")
    return clf


def get_texts(df):
    return df["combined_text"].fillna("").astype(str).tolist()


def get_sentiment_score(result):
    score_map = {item["label"]: item["score"] for item in result}
    return score_map["positive"] - score_map["negative"]


def add_sentiment_features(df, clf, name = "dataset", chunk_size = 100, batch_size = 16):
    df = df.copy()
    texts = get_texts(df)

    print(f"\nStarting sentiment scoring for {name}")
    print(f"{name} rows: {len(texts)}")

    all_results = []

    for start in range(0, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        print(f"{name}: scoring rows {start} to {end - 1}")
        batch_results = clf(
            texts[start:end],
            truncation = True,
            max_length = 300,
            top_k = None,
            batch_size = batch_size
            )
        
        all_results.extend(batch_results)
        print(f"{name}: finished rows {start} to {end - 1}.")

    print(f"{name}: attaching sentiment columns...")
    df["sentiment_score"] = [get_sentiment_score(r) for r in all_results]
    df["sentiment_strength"] = df["sentiment_score"].abs()
    print(f"{name}: done.")

    return df


def save_sentiment_table(df, filename):
    os.makedirs(OUTPUT_DIR, exist_ok = True)
    path = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(path, index = False)
    print(f"Saved: {path}")


def load_saved_sentiment_tables():
    cmv_path = os.path.join(OUTPUT_DIR, "CMV_sentiment.csv")
    omc_path = os.path.join(OUTPUT_DIR, "OMC_sentiment.csv")
    uo_path = os.path.join(OUTPUT_DIR, "UO_sentiment.csv")

    if all(os.path.exists(p) for p in [cmv_path, omc_path, uo_path]):
        print("Loading saved sentiment tables from outputs")
        CMV_sentiment = pd.read_csv(cmv_path)
        OMC_sentiment = pd.read_csv(omc_path)
        UO_sentiment = pd.read_csv(uo_path)
        return CMV_sentiment, OMC_sentiment, UO_sentiment
    
    return None


def make_all_sentiment_tables(use_saved = True):
    if use_saved:
        saved = load_saved_sentiment_tables()
        if saved is not None:
            return saved

    print("Building feature tables")
    CMV_features, OMC_features, UO_features = make_all_feature_tables()

    print("Feature tables built.")
    print("CMV feature shape:", CMV_features.shape)
    print("OMC feature shape:", OMC_features.shape)
    print("UO feature shape:", UO_features.shape)

    clf = load_sentiment_model()

    CMV_sentiment = add_sentiment_features(CMV_features, clf, name = "CMV")
    save_sentiment_table(CMV_sentiment, "CMV_sentiment.csv")

    OMC_sentiment = add_sentiment_features(OMC_features, clf, name = "OMC")
    save_sentiment_table(OMC_sentiment, "OMC_sentiment.csv")

    UO_sentiment = add_sentiment_features(UO_features, clf, name = "UO")
    save_sentiment_table(UO_sentiment, "UO_sentiment.csv")

    all_sentiment = pd.concat([CMV_sentiment, OMC_sentiment, UO_sentiment], ignore_index = True)
    save_sentiment_table(all_sentiment, "all_sentiment.csv")

    print("\nAll sentiment tables complete.")
    return CMV_sentiment, OMC_sentiment, UO_sentiment


if __name__ == "__main__":
    CMV_sentiment, OMC_sentiment, UO_sentiment = make_all_sentiment_tables(use_saved = False)

    preview_df(CMV_sentiment, "CMV sentiment features")
    preview_df(OMC_sentiment, "OMC sentiment features")
    preview_df(UO_sentiment, "UO sentiment features")

    print("\nCMV engagement counts")
    print(CMV_sentiment["engagement_level"].value_counts(dropna = False))

    print("\nOMC engagement counts")
    print(OMC_sentiment["engagement_level"].value_counts(dropna = False))

    print("\nUO engagement counts")
    print(UO_sentiment["engagement_level"].value_counts(dropna = False))