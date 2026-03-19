import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from statsmodels.miscmodels.ordinal_model import OrderedModel
from .evaluate import make_all_sentiment_tables


OUTPUT_DIR = "outputs"


def make_full_dataset():
    all_path = os.path.join(OUTPUT_DIR, "all_sentiment.csv")

    if os.path.exists(all_path):
        print("Loading full dataset from path outputs/all_sentiment.csv")
        df = pd.read_csv(all_path)
        return df

    print("Building dataset from evaluate.py")
    CMV_sentiment, OMC_sentiment, UO_sentiment = make_all_sentiment_tables(use_saved = True)
    df = pd.concat([CMV_sentiment, OMC_sentiment, UO_sentiment], ignore_index = True)
    return df


def prepare_data(df: pd.DataFrame):
    model_df = df[[
        "subreddit",
        "sentiment_score",
        "sentiment_strength",
        "engagement_level"
        ]].copy()

    model_df = model_df.dropna()

    model_df["sentiment_score"] = pd.to_numeric(model_df["sentiment_score"])
    model_df["sentiment_strength"] = pd.to_numeric(model_df["sentiment_strength"])
    model_df["engagement_level"] = pd.to_numeric(model_df["engagement_level"]).astype(int)

    X = pd.get_dummies(
        model_df[["subreddit", "sentiment_score", "sentiment_strength"]],
        columns = ["subreddit"],
        drop_first=True
        )

    X = X.astype(float)
    y = model_df["engagement_level"].astype(int)

    return X, y, model_df


def fit_ordinal_logit(X_train: pd.DataFrame, y_train: pd.Series):
    X_train = X_train.astype(float)
    y_train = y_train.astype(int)

    print("\nX dtypes:")
    print(X_train.dtypes)

    print("\ny dtype:")
    print(y_train.dtype)

    model = OrderedModel(
        endog = y_train,
        exog = X_train,
        distr = "logit"
        )


    result = model.fit(method = "bfgs", disp = False)
    return model, result


def predict_classes(result, X_test: pd.DataFrame):
    X_test = X_test.astype(float)
    pred_probs = result.model.predict(result.params, exog = X_test)
    pred_class = pred_probs.argmax(axis = 1)
    return pred_probs, pred_class


if __name__ == "__main__":
    print("Building full dataset")
    df = make_full_dataset()

    print("Full dataset shape:", df.shape)

    if "engagement_level" in df.columns:
        print("\nfull engagement counts:")
        print(df["engagement_level"].value_counts(dropna = False))

    X, y, model_df = prepare_data(df)

    print("Model dataframe shape:", model_df.shape)
    print(y.value_counts(dropna=False))

    print("\nTrain/test split")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size = 0.2,
        random_state = 42,
        stratify = y
        )

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    print("\nFitting ordinal logistic regression...")
    model, result = fit_ordinal_logit(X_train, y_train)

    print("\nMODEL SUMMARY")
    print(result.summary())

    print("\nMaking predictions...")
    pred_probs, y_pred = predict_classes(result, X_test)

    print("\nACCURACY")
    print(accuracy_score(y_test, y_pred))

    print("\nCLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred, labels = [0, 1, 2], zero_division = 0))

    print("\nCONFUSION MATRIX")
    print(confusion_matrix(y_test, y_pred, labels = [0, 1, 2]))

    print("\nFIRST 10 PREDICTIONS")
    comparison = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": y_pred
        })
    
    print(comparison.head(10))