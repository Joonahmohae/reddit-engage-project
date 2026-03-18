from transformers import pipeline

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

texts = [
    "I love this project",
    "This is awful",
    "This is okay"
    ]

results = clf(texts, truncation = True, max_length = 256)

for text, result in zip(texts, results):
    print(text, "->", result)