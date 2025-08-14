import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load datasets
df_fake = pd.read_csv("fakenews.csv")
df_real = pd.read_csv("truenews.csv")

# Add labels
df_fake['label'] = 1   # 1 = Fake
df_real['label'] = 0   # 0 = Real

# Combine into one DataFrame for better usage 
df = pd.concat([df_fake, df_real], axis=0).reset_index(drop=True)

#  drop 'subject' or 'date' cause we don't need them fro this classification, its totally useless here
df = df.drop(columns=['subject', 'date'], errors='ignore')


# Combine title + text for simplicity and better context
df['content'] = df['title'] + " " + df['text']

# Text cleaning: removing URLs and stopwords
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['content'] = df['content'].astype(str).apply(clean_text)

"""
training the data, using stratify, in order to select parts of the dataset that actually contains releveant info.
And transforming using tfidf vectorizer for transforming of the texts to numerics, in order to be fed into the model
"""

X_train, X_test, y_train, y_test = train_test_split(
    df['content'], df['label'],
    test_size=0.2, random_state=42, stratify=df['label']
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#here the training of the models takes place 
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

