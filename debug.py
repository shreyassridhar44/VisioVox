import pandas as pd

df = pd.read_csv('D:/visiovox_data/devA_manifest.csv')

print('Before:', repr(df['audio_path'].iloc[0]))
print('Total rows:', len(df))
print('NaN in lips_dir:', df['lips_dir'].isna().sum())
print('NaN in audio_path:', df['audio_path'].isna().sum())

# Fix audio_path
df['audio_path'] = df['audio_path'].apply(
    lambda x: x.replace('E:\\visiovox\\data\\devA_processed',
                         'D:\\visiovox_data\\devA_processed')
    if isinstance(x, str) else x
)

# Fix lips_dir
df['lips_dir'] = df['lips_dir'].apply(
    lambda x: x.replace('E:\\visiovox\\data\\devA_processed',
                         'D:\\visiovox_data\\devA_processed')
    if isinstance(x, str) else x
)

# Drop rows with NaN paths
before = len(df)
df = df.dropna(subset=['audio_path', 'lips_dir'])
after = len(df)
print(f'Dropped {before - after} rows with missing paths')

print('After:', repr(df['audio_path'].iloc[0]))
print('After lips:', repr(df['lips_dir'].iloc[0]))

df.to_csv('D:/visiovox_data/devA_manifest.csv', index=False)
print('Saved. Final rows:', len(df))