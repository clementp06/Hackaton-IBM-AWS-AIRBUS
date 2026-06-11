import pandas as pd
import numpy as np

v1 = pd.read_csv('submission_hybrid_tabpfn.csv')
v2 = pd.read_csv('submission_hybrid_optimized_v2.csv')

print('=' * 80)
print('COMPARAISON V1 vs V2')
print('=' * 80)

print('\nV1 (poids manuels 60/40):')
print(f'  Moyenne: {v1["corrosion_risk"].mean():.4f}')
print(f'  Std:     {v1["corrosion_risk"].std():.4f}')
print(f'  Min:     {v1["corrosion_risk"].min():.4f}')
print(f'  Max:     {v1["corrosion_risk"].max():.4f}')

print('\nV2 (poids optimisés par CV: 0% Iso / 100% Sig):')
print(f'  Moyenne: {v2["corrosion_risk"].mean():.4f}')
print(f'  Std:     {v2["corrosion_risk"].std():.4f}')
print(f'  Min:     {v2["corrosion_risk"].min():.4f}')
print(f'  Max:     {v2["corrosion_risk"].max():.4f}')

print('\nDifférences:')
diff = np.abs(v1['corrosion_risk'] - v2['corrosion_risk'])
print(f'  Différence absolue moyenne: {diff.mean():.4f}')
print(f'  Différence absolue max:     {diff.max():.4f}')
print(f'  Corrélation:                {v1["corrosion_risk"].corr(v2["corrosion_risk"]):.4f}')

print('\n' + '=' * 80)
print('INSIGHT CLÉ:')
print('=' * 80)
print('La validation croisée stratifiée a révélé que:')
print('  - Sigmoid calibration seule: Brier = 0.0678 (MEILLEUR)')
print('  - Isotonic calibration seule: Brier = 0.0690')
print('  - Mélange 50/50: Brier = 0.0683')
print('\nConclusion: Le modèle Sigmoid calibré est optimal pour ce problème.')
print('=' * 80)

# Made with Bob
