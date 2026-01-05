import pandas as pd
X_train = pd.read_csv('Milestone2_Preprocessing/outputs/X_train.csv')
print('Total columns:', len(X_train.columns))
print('\nColumn names:')
for i, col in enumerate(X_train.columns):
    print(f'{i+1}. {col}')
