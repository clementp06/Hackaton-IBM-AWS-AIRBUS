import argparse
import csv
import json
import random
from datetime import date
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
DEFAULT_CORROSIONS_PATH = DATA_DIR / "corrosions_training.csv"
DEFAULT_DISTRIBUTION_PATH = DATA_DIR / "best_age_distribution.json"
DEFAULT_OUTPUT_PATH = DATA_DIR / "prediction_training.csv"
DEFAULT_MIN_PREDICTIONS = 1
DEFAULT_MAX_PREDICTIONS = 8


def add_months(start_date, months):
    total_months = start_date.year * 12 + start_date.month - 1 + months
    year = total_months // 12
    month = total_months % 12 + 1
    return date(year, month, 1)


def parse_observation_date(value):
    year, month, day = value.split("-")
    return date(int(year), int(month), int(day))


def load_empirical_distribution(distribution_path):
    with distribution_path.open(encoding="utf-8") as file:
        report = json.load(file)

    best_distribution = report["best_distribution"]
    if best_distribution["name"] != "empirical_discrete":
        raise ValueError(
            "La meilleure loi du JSON n'est pas empirical_discrete. "
            "Relance test_distributions.py ou adapte ce generateur."
        )

    params = best_distribution["params"]
    values = [int(round(value)) for value in params["values"]]

    if "probabilities" in params:
        weights = [float(probability) for probability in params["probabilities"]]
    else:
        weights = [int(count) for count in params["counts"]]

    return values, weights


def weighted_sample_without_replacement(rng, values, weights, k):
    available_values = list(values)
    available_weights = list(weights)
    sampled_values = []

    if k > len(available_values):
        raise ValueError(
            f"Impossible de tirer {k} dates differentes avec seulement "
            f"{len(available_values)} ages possibles."
        )

    for _ in range(k):
        selected_value = rng.choices(available_values, weights=available_weights, k=1)[0]
        selected_index = available_values.index(selected_value)
        sampled_values.append(selected_value)
        del available_values[selected_index]
        del available_weights[selected_index]

    return sampled_values


def build_prediction_training(
    corrosions_path,
    distribution_path,
    seed,
    min_predictions,
    max_predictions,
):
    rng = random.Random(seed)
    age_values, age_weights = load_empirical_distribution(distribution_path)
    rows = []

    with corrosions_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for corrosion_row in reader:
            aircraft_id = corrosion_row["aircraft_id"]
            delivery_date = date(
                int(corrosion_row["aircraft_delivery_year"]),
                int(corrosion_row["aircraft_delivery_month"]),
                1,
            )
            observation_date = parse_observation_date(corrosion_row["observation_date"])

            prediction_count = rng.randint(min_predictions, max_predictions)
            sampled_ages = weighted_sample_without_replacement(
                rng,
                age_values,
                age_weights,
                prediction_count,
            )

            for age_months in sampled_ages:
                prediction_date = add_months(delivery_date, age_months)
                corrosion_risk = 0 if observation_date > prediction_date else 1

                rows.append(
                    {
                        "id": f"{aircraft_id}_{prediction_date:%Y-%m}",
                        "corrosion_risk": corrosion_risk,
                    }
                )

    return rows


def write_prediction_training(rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "corrosion_risk"])
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, output_path):
    positives = sum(row["corrosion_risk"] for row in rows)
    negatives = len(rows) - positives

    print(f"Fichier cree : {output_path}")
    print(f"Lignes : {len(rows)}")
    print(f"corrosion_risk=0 : {negatives}")
    print(f"corrosion_risk=1 : {positives}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Cree prediction_training.csv a partir de corrosions_training.csv "
            "en tirant des dates selon la loi empirique des dates de prediction."
        )
    )
    parser.add_argument("--corrosions", type=Path, default=DEFAULT_CORROSIONS_PATH)
    parser.add_argument("--distribution", type=Path, default=DEFAULT_DISTRIBUTION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-predictions", type=int, default=DEFAULT_MIN_PREDICTIONS)
    parser.add_argument("--max-predictions", type=int, default=DEFAULT_MAX_PREDICTIONS)
    args = parser.parse_args()

    if args.min_predictions < 1:
        raise ValueError("--min-predictions doit etre superieur ou egal a 1.")
    if args.max_predictions < args.min_predictions:
        raise ValueError("--max-predictions doit etre >= --min-predictions.")

    rows = build_prediction_training(
        args.corrosions,
        args.distribution,
        args.seed,
        args.min_predictions,
        args.max_predictions,
    )
    write_prediction_training(rows, args.output)
    print_summary(rows, args.output)


if __name__ == "__main__":
    main()
