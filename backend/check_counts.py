import pandas as pd
df = pd.read_csv('/home/robertom/Programs/SecureWaterTreatmentSystem/SWATDatasets/normal.csv', usecols=['MV101', 'P101', 'P102', 'MV201'])
print("MV101 counts:", df['MV101'].value_counts(dropna=False))
print("P101 counts:", df['P101'].value_counts(dropna=False))
print("P102 counts:", df['P102'].value_counts(dropna=False))
