"""
1. keep OMC, UO, and CMV seperate
2. Cleaning
- Remove duplicate posts using ID
- Remove deleted posts
- Remove posts without self text (is_self == FALSE)
- Remove is_video, url, and think about created_utc because posting time affects engagment alot. 
"""

import pandas as pd
from .collect import load_OMC_df, load_UO_df, load_CMV_df

loaded_OMC = load_OMC_df()
loaded_UO = load_UO_df()
loaded_CMV = load_CMV_df()

KEEP_COLS = [
    "id",
    "subreddit",
    "subreddit_name_prefixed",
    "created_utc",
    "title",
    "selftext",
    "score",
    "num_comments",
    "upvote_ratio",
    "is_self",
    "is_video",
    "url"
]   

def preview_df(df: pd.DataFrame, name: str, columns: list[str]) -> None:
    print(f"\n{name} shape:", df.shape)

    if df.empty:
        print(f"{name} is empty")
        return

    selected = [c for c in columns if c in df.columns]
    
    print(df[selected].head(10))

preview_df(loaded_OMC, "raw_OMC_df", KEEP_COLS)
preview_df(loaded_UO, "raw_UO_df", KEEP_COLS)
preview_df(loaded_CMV, "raw_CMVc_df", KEEP_COLS)




