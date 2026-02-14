from data_loader import DataLoader
from data_cleaner import DataCleaner

# Load data
loader = DataLoader("sample_customer_data.csv")
df = loader.load_csv()

print("\nOriginal Data:")
print(df.head())

# Clean data
cleaner = DataCleaner(df)
clean_df = cleaner.clean()

print("\nCleaned Data:")
print(clean_df.head())

