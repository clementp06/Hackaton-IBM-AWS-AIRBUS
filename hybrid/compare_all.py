import pandas as pd
import numpy as np

files = [
    'submission_hybrid_tabpfn.csv',
    'submission_hybrid_optimized_v2.csv', 
    'submission_final_ultra_optimized.csv',
    'submission_simple_lgb.csv'
]

print('=' * 80)
print('COMPARAISON DES 4 SOUMISSIONS')
print('=' * 80)

for f in files:
    df = pd.read_csv(f)
    pred = df['corrosion_risk']
    print(f'\n{f}:')
    print(f'  Moyenne: {pred.mean():.4f}')
    print(f'  Std:     {pred.std():.4f}')
    print(f'  Min:     {pred.min():.4f}')
    print(f'  Max:     {pred.max():.4f}')

print('\n' + '=' * 80)
print('RECOMMANDATION')
print('=' * 80)
print('\nLes 4 soumissions sont prêtes à tester.')
print('Soumettre dans cet ordre:')
print('  1. submission_final_ultra_optimized.csv (ensemble 3 modèles)')
print('  2. submission_simple_lgb.csv (LightGBM simple)')
print('  3. submission_hybrid_optimized_v2.csv (poids CV optimisés)')
print('=' * 80)

# Made with Bob
