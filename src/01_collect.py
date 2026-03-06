"""
1. Collect JSON files in POSTS
2. Load them into DF
3. 
"""

from pathlib import Path
import json
import pandas as pd

OMC_DIR = Path("data/raw/OMC")
UO_DIR = Path("data/raw/UO")
CMV_DIR = Path("data/raw/CMV")

# 1000 OMC posts
OMC_POSTS = [
    OMC_DIR/ "OMC-hot-first-100.json",
    OMC_DIR/ "OMC-hot-second-100.json",
    OMC_DIR/ "OMC-hot-third-100.json",
    OMC_DIR/ "OMC-hot-fourth-100.json",
    OMC_DIR/ "OMC-hot-fifth-100.json",

    OMC_DIR/ "OMC-new-first-100.json",
    OMC_DIR/ "OMC-new-second-100.json",
    OMC_DIR/ "OMC-new-third-100.json",
    OMC_DIR/ "OMC-new-fourth-100.json",
    OMC_DIR/ "OMC-new-fifth-100.json"
]

# 1000 UO posts
UO_POSTS = [
    UO_DIR / "UO-hot-first-100.json",
    UO_DIR / "UO-hot-second-100.json",
    UO_DIR / "UO-hot-third-100.json",
    UO_DIR / "UO-hot-fourth-100.json",
    UO_DIR / "UO-hot-fifth-100.json",

    UO_DIR / "UO-new-first-100.json",
    UO_DIR / "UO-new-second-100.json",
    UO_DIR / "UO-new-third-100.json",
    UO_DIR / "UO-new-fourth-100.json",
    UO_DIR / "UO-new-fifth-100.json",
]

# 1000 CMV posts
CMV_POSTS = [
    CMV_DIR / "CMV-hot-first-100.json",
    CMV_DIR / "CMV-hot-second-100.json",
    CMV_DIR / "CMV-hot-third-100.json",
    CMV_DIR / "CMV-hot-fourth-100.json",
    CMV_DIR / "CMV-hot-fifth-100.json",

    CMV_DIR / "CMV-new-first-100.json",
    CMV_DIR / "CMV-new-second-100.json",
    CMV_DIR / "CMV-new-third-100.json",
    CMV_DIR / "CMV-new-fourth-100.json",
    CMV_DIR / "CMV-new-fifth-100.json",
]

def jsons_to_df(paths: list[Path]) -> pd.DataFrame:
    frames = []
    
    for path in paths:
        if not path.exists():
            print(f"Skipping (not found): {path}")
            continue

        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # Reddit listing -> data -> children -> each child["data"] is one post
        children = obj.get("data", {}).get("children", [])

        rows = []

        for child in children:
            if not isinstance(child, dict):
                continue
            post = child.get("data", {})
            if isinstance(post, dict):
                rows.append(post)

        if not rows:
            print(f"No posts found in: {path}")
            continue

        temp = pd.DataFrame(rows)
        temp["source_file"] = path.name  
        frames.append(temp)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index = True)

raw_OMC_df = jsons_to_df(OMC_POSTS)
raw_UO_df = jsons_to_df(UO_POSTS)
raw_CMV_df = jsons_to_df(CMV_POSTS)

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

preview_df(raw_OMC_df, "raw_OMC_df", KEEP_COLS)
preview_df(raw_UO_df, "raw_UO_df", KEEP_COLS)
preview_df(raw_CMV_df, "raw_CMV_df", KEEP_COLS)
