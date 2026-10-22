import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def main():
    print("Loading datasets...")
    real_news = pd.read_csv('PolitiFact_real_news_content.csv')
    fake_news = pd.read_csv('PolitiFact_fake_news_content.csv')

    print("Adding label columns...")
    real_news['label'] = 'REAL'
    fake_news['label'] = 'FAKE'

    print("Combining datasets...")
    news_df = pd.concat([real_news, fake_news], ignore_index=True)

    print("Preprocessing data...")
    news_df['content'] = news_df['title'] + " " + news_df['text']

    X = news_df['content']
    y = news_df['label']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))

    print("Saving model and vectorizer...")
    joblib.dump(model, 'logistic_regression_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    print("Model and vectorizer saved to disk.")

if __name__ == "__main__":
    main()
