import pandas as pd
from .collect import load_OMC_df, load_UO_df, load_CMV_df


KEEP_COLS = [
    "id",
    "subreddit",
    "title",
    "selftext",
    "score",
    "num_comments",
    "upvote_ratio",
    "created_utc"
    ]  


def clean_posts(df: pd.DataFrame, keep_columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    # Basic text prep
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["selftext"] = df["selftext"].fillna("").astype(str).str.strip()

    # Keep only self posts
    if "is_self" in df.columns:
        df = df[df["is_self"]]

    # Remove deleted/removed posts
    df = df[~df["selftext"].isin(["[deleted]", "[removed]", ""])]
    df = df[~df["title"].isin(["[deleted]", "[removed]"])]

    # Remove duplicate content
    df = df.drop_duplicates(subset = ["title", "selftext"])

    # Clean only text columns
    for col in ["title", "selftext"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .str.replace(r"\\n|\\t|\\r", " ", regex = True)
                .str.replace(r"[\n\t\r]", " ", regex = True)
                .str.replace(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", regex = True)
                .str.replace(r"http\S+|www\.\S+", " ", regex = True)
                .str.replace(r"\\([\-*_[\]()])", r"\1", regex = True)
                .str.replace(r"\s+", " ", regex = True)
                .str.strip()
                )
            
    if "title" in df.columns:
        df["title"] = df["title"].str.replace(r"^\s*CMV\s*:\s*|^\s*OMC\s*:\s*|^\s*UO\s*:\s*", "", regex = True, case = False)

    # Keep requested columns
    df = df[[c for c in keep_columns if c in df.columns]]

    return df


def preview_df(df: pd.DataFrame, name: str) -> None:
    print(f"\n{name} shape:", df.shape)

    if df.empty:
        print(f"{name} is empty")
        return
    
    print(df.head(10))


def get_cleaned_OMC_df() -> pd.DataFrame:
    loaded_OMC = load_OMC_df()
    return clean_posts(loaded_OMC, KEEP_COLS)


def get_cleaned_UO_df() -> pd.DataFrame:
    loaded_UO = load_UO_df()
    return clean_posts(loaded_UO, KEEP_COLS)


def get_cleaned_CMV_df() -> pd.DataFrame:
    loaded_CMV = load_CMV_df()
    return clean_posts(loaded_CMV, KEEP_COLS)


# compares number of posts before and after cleaning
if __name__ == "__main__":
    loaded_OMC = load_OMC_df()
    cleaned_OMC = clean_posts(loaded_OMC, KEEP_COLS)
    preview_df(cleaned_OMC, "cleaned_OMC_df")
    #preview_df(loaded_OMC, "loaded_OMC_df")

    loaded_UO = load_UO_df()
    cleaned_UO = clean_posts(loaded_UO, KEEP_COLS)
    preview_df(cleaned_UO, "cleaned_UO_df")
    #preview_df(loaded_UO, "loaded_UO_df")

    loaded_CMV = load_CMV_df()
    cleaned_CMV = clean_posts(loaded_CMV, KEEP_COLS)
    preview_df(cleaned_CMV, "cleaned_CMV_df")
    #preview_df(loaded_CMV, "loaded_CMV_df")





