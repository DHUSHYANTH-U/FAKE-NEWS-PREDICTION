import pandas as pd

real_news = pd.read_csv('PolitiFact_real_news_content.csv')
fake_news = pd.read_csv('PolitiFact_fake_news_content.csv')

print("Real news samples:", len(real_news))
print("Fake news samples:", len(fake_news))
