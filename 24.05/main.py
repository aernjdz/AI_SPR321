import pandas as pd
from textblob import TextBlob

# === 1. Завантаження CSV-файлу ===
df = pd.read_csv("text.csv")  
# === 2. Функція для аналізу тональності ===
def classify_sentiment(text):
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

# === 3. Застосування моделі ===
df['Sentiment'] = df['Message'].apply(classify_sentiment)

# === 4. Підрахунок результатів ===
result = df['Sentiment'].value_counts()

print("Статистика аналізу тональності:")
print(result)

# (Опційно) Зберегти в CSV
df.to_csv("analyzed_feedback.csv", index=False)
