from sklearn.preprocessing import LabelEncoder
import pandas as pd

print(chr(27) + '[2J')

df = pd.read_csv('Marks.csv')
sub = LabelEncoder()

df['sub_encoded'] = sub.fit_transform(df['subject'].values)
print(df)

# sub_encoded to be added & created new file
# df.to_csv('Datasets/marks_encoded.csv')
