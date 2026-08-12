import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the CSV file
csv_path = "/path/to/SEVA/Emotion-FAN/csv/crema/fps25.csv"

# Read the data
df = pd.read_csv(csv_path)

# Split 9:1 (fixing random_state keeps the split identical every time)
train_df, test_df = train_test_split(df, test_size=0.1, shuffle=True, random_state=42)

# Save the results
train_df.to_csv("/path/to/SEVA/Emotion-FAN/csv/crema/train.csv", index=False)
test_df.to_csv("/path/to/SEVA/Emotion-FAN/csv/crema/test.csv", index=False)
