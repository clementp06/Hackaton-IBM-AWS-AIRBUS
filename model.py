import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# 1. Données : environnement + label (date, aircraft_id, corroded)
env    = pd.read_csv('data/environment_training.csv')
labels = pd.read_csv('data/corrosion_labels.csv')

env['date']    = pd.to_datetime(env['month_start_date']).dt.to_period('M').astype(str)
labels['date'] = pd.to_datetime(labels['date']).dt.to_period('M').astype(str)

df = env.merge(labels[['aircraft_id', 'date', 'corroded']], on=['aircraft_id', 'date'])

# 2. Features (toutes les colonnes numériques) + cible
X = df.drop(columns=['aircraft_id', 'year_month', 'month_start_date', 'date', 'corroded'])
X = X.select_dtypes('number')
y = df['corroded']

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Modèle
model = xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# 4. Probabilité de corrosion (%)
proba = model.predict_proba(X_test)[:, 1]
print("AUC :", round(roc_auc_score(y_test, proba), 4))
print("Probas de corrosion (%) :", (proba[:5] * 100).round(1))
