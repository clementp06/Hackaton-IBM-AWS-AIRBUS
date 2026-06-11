# Script PowerShell pour lancer l'optimisation complete de l'ensemble
# Temps estime: 3-4 heures

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "FULL ENSEMBLE OPTIMIZATION PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Etape 1: Baseline avec feature selection
Write-Host "[1/3] Running baseline with feature selection..." -ForegroundColor Yellow
uv run python train_ensemble.py `
  --use-feature-selection `
  --top-n-features 50 `
  --min-gain 800 `
  --corr-threshold 0.95

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Baseline training failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Baseline complete!" -ForegroundColor Green
Write-Host ""

# Etape 2: Optimisation des hyperparametres
Write-Host "[2/3] Optimizing hyperparameters (this will take 2-3 hours)..." -ForegroundColor Yellow
uv run python train_ensemble.py `
  --use-feature-selection `
  --optimize-hyperparams `
  --n-trials 100 `
  --save-params best_params.json

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Hyperparameter optimization failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Hyperparameter optimization complete!" -ForegroundColor Green
Write-Host ""

# Etape 3: Optimisation des poids avec les meilleurs parametres
Write-Host "[3/3] Optimizing ensemble weights with best parameters..." -ForegroundColor Yellow
uv run python train_ensemble.py `
  --use-feature-selection `
  --load-params best_params.json `
  --optimize-weights

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Weight optimization failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "OPTIMIZATION COMPLETE!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Results saved in:"
Write-Host "  - ensemble_results.json"
Write-Host "  - best_params.json"
Write-Host ""
Write-Host "To compare results:"
Write-Host "  uv run python compare_results.py"

# Made with Bob
