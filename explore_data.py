import pandas as pd

def main():
    # Load real news dataset
    real_news = pd.read_csv('PolitiFact_real_news_content.csv')

    # Load fake news dataset
    fake_news = pd.read_csv('PolitiFact_fake_news_content.csv')

    # Add label columns
    real_news['label'] = 'REAL'
    fake_news['label'] = 'FAKE'

    # Combine both datasets
    news_df = pd.concat([real_news, fake_news], ignore_index=True)

    # Show combined dataset shape
    print("Combined dataset shape:", news_df.shape)

    # Show first 5 rows to understand structure
    print(news_df.head())

    # Check for missing values per column
    print("\nMissing values by column:")
    print(news_df.isnull().sum())

if __name__ == "__main__":
    main()
