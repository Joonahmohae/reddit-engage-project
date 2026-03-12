from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# Initialize once (reuse for all rows/texts)
analyzer = SentimentIntensityAnalyzer()

# Example texts
texts = [
    "for those who are into kink, how did you realize you were into it?"
]

# Score each text
results = []
for text in texts:
    scores = analyzer.polarity_scores(text)  # returns neg, neu, pos, compound

    # Simple label based on compound score
    if scores["compound"] >= 0.05:
        label = "positive"
    elif scores["compound"] <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    results.append({
        "text": text,
        "neg": scores["neg"],
        "neu": scores["neu"],
        "pos": scores["pos"],
        "compound": scores["compound"],
        "sentiment_label": label
    })

df = pd.DataFrame(results)
print(df)