"""
1. Collect JSON files in POSTS
2. Load them into DF
3. 
"""

from pathlib import Path
import json
import pandas as pd

DATA_DIR = Path("data/raw/OMC")

OMC_POSTS = [
    DATA_DIR/ "OMC-hot-first-100.json",
    DATA_DIR/ "OMC-hot-second-100.json",
    DATA_DIR/ "OMC-hot-third-100.json",
    DATA_DIR/ "OMC-hot-fourth-100.json",
    DATA_DIR/ "OMC-hot-fifth-100.json",

    DATA_DIR/ "OMC-new-first-100.json",
    DATA_DIR/ "OMC-new-second-100.json",
    DATA_DIR/ "OMC-new-third-100.json",
    DATA_DIR/ "OMC-new-fourth-100.json",
    DATA_DIR/ "OMC-new-fifth-100.json",

    DATA_DIR/ "OMC-topweek-first-100.json",
    DATA_DIR/ "OMC-topweek-second-100.json",
    DATA_DIR/ "OMC-topweek-third-100.json",
    DATA_DIR/ "OMC-topweek-fourth-100.json",
    DATA_DIR/ "OMC-topweek-fifth-100.json"
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

print("confirmed")
print(raw_OMC_df.shape)
print(raw_OMC_df[["id", "subreddit", ]].head(10))
print(raw_OMC_df[["title", "selftext"]].head(10))
print(raw_OMC_df[["score", "num_comments"]].head(10))
print(raw_OMC_df[["upvote_ratio", "ups", "total_awards_received", "num_crossposts"]].head(10))