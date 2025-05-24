import pandas as pd
from textblob import TextBlob

df = pd.read_csv("text.csv")  

def classify_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['Message'].apply(classify_sentiment)

result = df['Sentiment'].value_counts()

print("Статистика аналізу тональності:")
print(result)

df.to_csv("analyzed_feedback.csv", index=False)
