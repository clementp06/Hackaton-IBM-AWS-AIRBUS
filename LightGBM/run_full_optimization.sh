#!/bin/bash
# Script pour lancer l'optimisation complete de l'ensemble
# Temps estime: 3-4 heures

echo "=========================================="
echo "FULL ENSEMBLE OPTIMIZATION PIPELINE"
echo "=========================================="
echo ""

# Etape 1: Baseline avec feature selection
echo "[1/3] Running baseline with feature selection..."
uv run python train_ensemble.py \
  --use-feature-selection \
  --top-n-features 50 \
  --min-gain 800 \
  --corr-threshold 0.95

if [ $? -ne 0 ]; then
    echo "ERROR: Baseline training failed"
    exit 1
fi

echo ""
echo "Baseline complete!"
echo ""

# Etape 2: Optimisation des hyperparametres
echo "[2/3] Optimizing hyperparameters (this will take 2-3 hours)..."
uv run python train_ensemble.py \
  --use-feature-selection \
  --optimize-hyperparams \
  --n-trials 100 \
  --save-params best_params.json

if [ $? -ne 0 ]; then
    echo "ERROR: Hyperparameter optimization failed"
    exit 1
fi

echo ""
echo "Hyperparameter optimization complete!"
echo ""

# Etape 3: Optimisation des poids avec les meilleurs parametres
echo "[3/3] Optimizing ensemble weights with best parameters..."
uv run python train_ensemble.py \
  --use-feature-selection \
  --load-params best_params.json \
  --optimize-weights

if [ $? -ne 0 ]; then
    echo "ERROR: Weight optimization failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "OPTIMIZATION COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - ensemble_results.json"
echo "  - best_params.json"
echo ""
echo "To compare results:"
echo "  uv run python compare_results.py"

# Made with Bob
