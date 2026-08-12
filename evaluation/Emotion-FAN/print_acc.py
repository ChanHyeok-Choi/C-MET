import pandas as pd

# Path to the CSV file
csv_path = "/path/to/SEVA/Emotion-FAN/csv/crema/test/test.csv"

# Read the CSV file
df = pd.read_csv(csv_path)

# Group by emotion and intensity, then compute accuracy
result = (
    df.groupby(['gt_emotion', 'intensity'])
    .apply(lambda x: (x['gt_emotion'] == x['predicted_emotion']).mean())
    .reset_index(name='accuracy')
)

print(result)
