import pandas as pd

def main():
    # Load datasets
    real_news = pd.read_csv('PolitiFact_real_news_content.csv')
    fake_news = pd.read_csv('PolitiFact_fake_news_content.csv')

    # Add label columns
    real_news['label'] = 'REAL'
    fake_news['label'] = 'FAKE'

    # Combine datasets
    news_df = pd.concat([real_news, fake_news], ignore_index=True)

    # Analyze class distribution
    class_distribution = news_df['label'].value_counts()

    print("Class distribution in dataset:")
    print(class_distribution)

if __name__ == "__main__":
    main()
