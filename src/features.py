import pandas as pd
import numpy as np
import re
from .clean import get_cleaned_OMC_df, get_cleaned_UO_df, get_cleaned_CMV_df, preview_df


cleaned_OMC = get_cleaned_OMC_df()
cleaned_UO = get_cleaned_UO_df()
cleaned_CMV = get_cleaned_CMV_df()


# converts created_utc to datetime form then subtract it from scrape time UTC to create a post_age column. Later convert this to hours so it gives us the hours past since the post was created
def add_post_utc_age(df: pd.DataFrame, column: str, scrape_time_utc: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    df["created_utc_datetime"] = pd.to_datetime(df[column], unit = "s", utc = True)
    df["post_age"] = scrape_time_utc - df["created_utc_datetime"]
    df["post_age_hours"] = df["post_age"].dt.total_seconds() / 3600
    return df


# combines text and self text to one big chunk
def combine_texts(df: pd.DataFrame, column_1: str, column_2: str) -> pd.DataFrame:
    df = df.copy()
    df["combined_text"] = df[column_1] + " " + df[column_2]
    return df


# creates comment_rate which shows the number of comments a post recived relative to the post age. Log transform to adjust skweness
def add_comment_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["post_age_hours"] > 0]
    df["comment_rate"] = df["num_comments"] / df["post_age_hours"]
    df["log_comment_rate"] = np.log1p(df["comment_rate"])
    return df


# create engagment level using quantiles
def add_engagement_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["engagement_level"] = pd.qcut(df["log_comment_rate"], q = 3, labels = [0, 1, 2])
    return df


def make_feature_table(df: pd.DataFrame, scrape_time_utc: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()

    df = add_post_utc_age(df, "created_utc", scrape_time_utc)
    df = combine_texts(df, "title", "selftext")
    df = add_comment_rate(df)
    df = add_engagement_level(df)

    return df[[
        "id",
        "subreddit",
        "combined_text",
        "post_age_hours",
        "num_comments",
        "comment_rate",
        "log_comment_rate",
        "engagement_level"
        ]]


def make_all_feature_tables():
    CMV_scrape_time = pd.Timestamp("2026-03-06 01:22:18", tz="UTC")
    OMC_scrape_time = pd.Timestamp("2026-03-04 07:20:18", tz="UTC")
    UO_scrape_time = pd.Timestamp("2026-03-06 00:27:12", tz="UTC")

    CMV_features = make_feature_table(cleaned_CMV, CMV_scrape_time)
    OMC_features = make_feature_table(cleaned_OMC, OMC_scrape_time)
    UO_features = make_feature_table(cleaned_UO, UO_scrape_time)

    return CMV_features, OMC_features, UO_features

    



