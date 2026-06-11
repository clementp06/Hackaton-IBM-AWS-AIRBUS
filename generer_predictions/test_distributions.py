import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "prediction_age_distribution.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "best_age_distribution.json"


@dataclass
class Candidate:
    name: str
    sampler: callable
    params: dict


def read_ages(path):
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return np.array([float(row["age_months"]) for row in reader])


def percentile_rmse(reference, sample):
    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
    ref_q = np.percentile(reference, percentiles)
    sample_q = np.percentile(sample, percentiles)
    return float(np.sqrt(np.mean((ref_q - sample_q) ** 2)))


def histogram_distance(reference, sample):
    min_value = min(reference.min(), sample.min())
    max_value = max(reference.max(), sample.max())
    bins = np.arange(math.floor(min_value), math.ceil(max_value) + 2)
    ref_hist, _ = np.histogram(reference, bins=bins, density=True)
    sample_hist, _ = np.histogram(sample, bins=bins, density=True)
    return float(np.mean(np.abs(ref_hist - sample_hist)))


def score_sample(reference, sample):
    ks = stats.ks_2samp(reference, sample).statistic
    wasserstein = stats.wasserstein_distance(reference, sample)
    q_rmse = percentile_rmse(reference, sample)
    hist = histogram_distance(reference, sample)

    return {
        "ks": float(ks),
        "wasserstein": float(wasserstein),
        "quantile_rmse": float(q_rmse),
        "histogram_l1": float(hist),
    }


def fit_candidates(ages):
    min_age = float(ages.min())
    max_age = float(ages.max())
    mean_age = float(ages.mean())
    std_age = float(ages.std(ddof=0))
    span = max_age - min_age
    empirical_values, empirical_counts = np.unique(ages, return_counts=True)
    empirical_probabilities = empirical_counts / empirical_counts.sum()

    candidates = []

    candidates.append(
        Candidate(
            name="empirical_discrete",
            sampler=lambda rng, size: rng.choice(
                empirical_values,
                size=size,
                replace=True,
                p=empirical_probabilities,
            ),
            params={
                "values": [float(value) for value in empirical_values],
                "counts": [int(count) for count in empirical_counts],
                "probabilities": [
                    float(probability) for probability in empirical_probabilities
                ],
            },
        )
    )

    candidates.append(
        Candidate(
            name="normal_clipped",
            sampler=lambda rng, size: np.clip(
                rng.normal(mean_age, std_age, size=size),
                min_age,
                max_age,
            ),
            params={"mean": mean_age, "std": std_age, "min": min_age, "max": max_age},
        )
    )

    beta_loc = min_age - 1
    beta_scale = span + 2
    beta_alpha, beta_beta, _, _ = stats.beta.fit(
        ages,
        floc=beta_loc,
        fscale=beta_scale,
    )
    candidates.append(
        Candidate(
            name="beta_scaled",
            sampler=lambda rng, size: stats.beta.rvs(
                beta_alpha,
                beta_beta,
                loc=beta_loc,
                scale=beta_scale,
                size=size,
                random_state=rng,
            ),
            params={
                "alpha": float(beta_alpha),
                "beta": float(beta_beta),
                "loc": float(beta_loc),
                "scale": float(beta_scale),
            },
        )
    )

    gamma_shape, _, gamma_scale = stats.gamma.fit(ages, floc=0)
    candidates.append(
        Candidate(
            name="gamma",
            sampler=lambda rng, size: stats.gamma.rvs(
                gamma_shape,
                loc=0,
                scale=gamma_scale,
                size=size,
                random_state=rng,
            ),
            params={"shape": float(gamma_shape), "loc": 0.0, "scale": float(gamma_scale)},
        )
    )

    weibull_shape, _, weibull_scale = stats.weibull_min.fit(ages, floc=0)
    candidates.append(
        Candidate(
            name="weibull_min",
            sampler=lambda rng, size: stats.weibull_min.rvs(
                weibull_shape,
                loc=0,
                scale=weibull_scale,
                size=size,
                random_state=rng,
            ),
            params={
                "shape": float(weibull_shape),
                "loc": 0.0,
                "scale": float(weibull_scale),
            },
        )
    )

    lognorm_shape, _, lognorm_scale = stats.lognorm.fit(ages, floc=0)
    candidates.append(
        Candidate(
            name="lognormal",
            sampler=lambda rng, size: stats.lognorm.rvs(
                lognorm_shape,
                loc=0,
                scale=lognorm_scale,
                size=size,
                random_state=rng,
            ),
            params={
                "shape": float(lognorm_shape),
                "loc": 0.0,
                "scale": float(lognorm_scale),
            },
        )
    )

    candidates.append(
        Candidate(
            name="uniform",
            sampler=lambda rng, size: rng.uniform(min_age, max_age, size=size),
            params={"min": min_age, "max": max_age},
        )
    )

    return candidates


def evaluate_candidate(candidate, reference, n_tests, seed):
    rng = np.random.default_rng(seed)
    scores = []

    for _ in range(n_tests):
        sample = candidate.sampler(rng, len(reference))
        sample = np.rint(sample).astype(float)
        scores.append(score_sample(reference, sample))

    summary = {}
    for metric in scores[0]:
        values = np.array([score[metric] for score in scores])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "p95": float(np.percentile(values, 95)),
        }

    # Score composite simple: plus petit = plus fidele.
    composite = (
        summary["ks"]["mean"]
        + summary["wasserstein"]["mean"] / max(reference.std(ddof=0), 1)
        + summary["quantile_rmse"]["mean"] / max(reference.std(ddof=0), 1)
        + summary["histogram_l1"]["mean"]
    )

    return {
        "name": candidate.name,
        "params": candidate.params,
        "composite_score": float(composite),
        "metrics": summary,
    }


def save_report(output_path, ages, results):
    best = min(results, key=lambda result: result["composite_score"])
    report = {
        "best_distribution": best,
        "all_results": sorted(results, key=lambda result: result["composite_score"]),
        "reference": {
            "count": int(len(ages)),
            "min": float(ages.min()),
            "max": float(ages.max()),
            "mean": float(ages.mean()),
            "std": float(ages.std(ddof=0)),
            "percentiles": {
                str(percent): float(np.percentile(ages, percent))
                for percent in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
        },
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


def print_results(report):
    print("Classement des distributions, score plus petit = meilleur")
    print()

    for rank, result in enumerate(report["all_results"], start=1):
        metrics = result["metrics"]
        print(
            f"{rank}. {result['name']:<18} "
            f"score={result['composite_score']:.4f} "
            f"KS={metrics['ks']['mean']:.4f} "
            f"W={metrics['wasserstein']['mean']:.2f} "
            f"Q_RMSE={metrics['quantile_rmse']['mean']:.2f}"
        )

    print()
    print(f"Meilleure distribution : {report['best_distribution']['name']}")
    print("Parametres :")
    print(json.dumps(report["best_distribution"]["params"], indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Compare plusieurs lois pour reproduire la distribution des ages de prediction."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-tests", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ages = read_ages(args.input)
    candidates = fit_candidates(ages)

    results = [
        evaluate_candidate(candidate, ages, args.n_tests, args.seed + index)
        for index, candidate in enumerate(candidates)
    ]

    report = save_report(args.output, ages, results)
    print_results(report)
    print()
    print(f"Rapport complet ecrit dans : {args.output}")


if __name__ == "__main__":
    main()
