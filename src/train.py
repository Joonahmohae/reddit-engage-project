from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

sia = SentimentIntensityAnalyzer()

texts = [
    "example",
    "example",
    "example"
]

engagement = [120, 80, 25]

rows = []
for text in texts:
    s = sia.polarity_scores(text)
    rows.append({
        "neg": s["neg"],
        "neu": s["neu"],
        "pos": s["pos"],
        "compound": s["compound"],
        "length": len(text),
        "engagement": engagement[len(rows)]
    })

df = pd.DataFrame(rows)

X = df[["neg", "neu", "pos", "compound", "length"]]
y = df["engagement"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(preds)
print("Actual:", y_test.values)
print("Predicted:", preds)