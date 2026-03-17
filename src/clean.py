"""
1. keep OMC, UO, and CMV seperate
2. Cleaning
- Remove duplicate posts 
- Remove deleted posts
- Remove posts without self text (is_self == FALSE)
- drop columns is_video, url
- also drop created_utc because we are only concered about the text content, although the result is going to be more accurate with the time since time affects engagment 
"""


import pandas as pd
from .collect import load_OMC_df, load_UO_df, load_CMV_df


loaded_OMC = load_OMC_df()
loaded_UO = load_UO_df()
loaded_CMV = load_CMV_df()


KEEP_COLS = [
    "id",
    "subreddit",
    "title",
    "selftext",
    "score",
    "num_comments",
    "upvote_ratio",
    "is_self"
    ]  


def clean_posts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    
    df["title"] = df["title"].fillna("").str.strip()
    df["selftext"] = df["selftext"].fillna("").str.strip()

    # keep only self posts
    df = df[df["is_self"]]

    # remove deleted/removed posts
    df = df[~df["selftext"].isin(["[deleted]", "[removed]", ""])]
    df = df[~df["title"].isin(["[deleted]", "[removed]"])]

    # remove duplicate content
    df = df.drop_duplicates(subset=["title", "selftext"])

    df = df[[c for c in columns if c in df.columns]]
    
    return df

cleaned_OMC = clean_posts(loaded_OMC, KEEP_COLS)
cleaned_UO = clean_posts(loaded_UO, KEEP_COLS)
cleaned_CMV = clean_posts(loaded_CMV, KEEP_COLS)


def preview_df(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} shape:", df.shape)

    if df.empty:
        print(f"{name} is empty")
        return
    
    print(df.head(10))

preview_df(cleaned_OMC, "cleaned_OMC_df")
preview_df(loaded_OMC, "loaded_OMC_df")

preview_df(cleaned_UO, "cleaned_UO_df")
preview_df(loaded_UO, "loaded_UO_df")

preview_df(cleaned_CMV, "cleaned_CMVc_df")
preview_df(loaded_CMV, "loaded_CMVc_df")






