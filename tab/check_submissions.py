import pandas as pd

files = ['submission.csv', 'submission_enhanced.csv']

print('=' * 60)
print('SOUMISSIONS TABPFN DISPONIBLES')
print('=' * 60)

for f in files:
    try:
        df = pd.read_csv(f)
        pred = df['corrosion_risk']
        print(f'\n{f}:')
        print(f'  Shape: {df.shape}')
        print(f'  Moyenne: {pred.mean():.4f}')
        print(f'  Std: {pred.std():.4f}')
        print(f'  Min: {pred.min():.4f}')
        print(f'  Max: {pred.max():.4f}')
    except Exception as e:
        print(f'\n{f}: ERROR - {e}')

print('\n' + '=' * 60)
print('RECOMMANDATION: Utiliser submission_enhanced.csv')
print('=' * 60)

# Made with Bob
