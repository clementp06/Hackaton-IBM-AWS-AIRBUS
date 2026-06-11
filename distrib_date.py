import csv
from collections import Counter
from datetime import date
from pathlib import Path
from statistics import mean, median, pstdev


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
ENVIRONMENT_TEST_PATH = DATA_DIR / "environment_test.csv"


def parse_year_month(value):
    year, month = value[:7].split("-")
    return date(int(year), int(month), 1)


def month_delta(start, end):
    return (end.year - start.year) * 12 + (end.month - start.month)


def percentile(sorted_values, percent):
    if not sorted_values:
        return None

    index = (len(sorted_values) - 1) * percent / 100
    lower = int(index)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def read_aircraft_creation_dates(environment_path):
    creation_dates = {}

    with environment_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            aircraft_id = row["aircraft_id"]
            observed_date = parse_year_month(row["year_month"])

            if aircraft_id not in creation_dates or observed_date < creation_dates[aircraft_id]:
                creation_dates[aircraft_id] = observed_date

    return creation_dates


def read_prediction_dates(sample_submission_path):
    prediction_dates = []

    with sample_submission_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            aircraft_id, prediction_year_month = row["id"].rsplit("_", 1)
            prediction_dates.append(
                {
                    "aircraft_id": aircraft_id,
                    "prediction_date": parse_year_month(prediction_year_month),
                    "sample_id": row["id"],
                }
            )

    return prediction_dates


def build_distribution(sample_submission_path, environment_path):
    creation_dates = read_aircraft_creation_dates(environment_path)
    prediction_dates = read_prediction_dates(sample_submission_path)

    distribution = []
    missing_aircraft_ids = set()

    for prediction in prediction_dates:
        aircraft_id = prediction["aircraft_id"]
        creation_date = creation_dates.get(aircraft_id)

        if creation_date is None:
            missing_aircraft_ids.add(aircraft_id)
            continue

        age_months = month_delta(creation_date, prediction["prediction_date"])
        distribution.append(
            {
                "sample_id": prediction["sample_id"],
                "aircraft_id": aircraft_id,
                "creation_date": creation_date.isoformat(),
                "prediction_date": prediction["prediction_date"].isoformat(),
                "age_months": age_months,
            }
        )

    return distribution, missing_aircraft_ids


def print_distribution_summary(distribution, missing_aircraft_ids):
    ages = sorted(row["age_months"] for row in distribution)
    if not ages:
        print("Aucune ligne exploitable trouvee.")
        return

    print("Distribution de prediction_date - aircraft_creation_date")
    print(f"Lignes sample_submission exploitees : {len(ages)}")
    print(f"Avions absents de environment_test : {len(missing_aircraft_ids)}")
    print()
    print("Statistiques en mois")
    print(f"min    : {ages[0]:.0f}")
    print(f"p05    : {percentile(ages, 5):.1f}")
    print(f"p25    : {percentile(ages, 25):.1f}")
    print(f"median : {median(ages):.1f}")
    print(f"mean   : {mean(ages):.1f}")
    print(f"p75    : {percentile(ages, 75):.1f}")
    print(f"p95    : {percentile(ages, 95):.1f}")
    print(f"max    : {ages[-1]:.0f}")
    print(f"std    : {pstdev(ages):.1f}")
    print()
    print("Histogramme par age en mois")

    for age_months, count in sorted(Counter(ages).items()):
        print(f"{age_months:>3} mois : {count}")

    if missing_aircraft_ids:
        preview = ", ".join(sorted(missing_aircraft_ids)[:10])
        print()
        print(f"Avions manquants, apercu : {preview}")


def write_distribution(distribution, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "sample_id",
            "aircraft_id",
            "creation_date",
            "prediction_date",
            "age_months",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(distribution)


def main():
    distribution, missing_aircraft_ids = build_distribution(
        SAMPLE_SUBMISSION_PATH,
        ENVIRONMENT_TEST_PATH,
    )
    print_distribution_summary(distribution, missing_aircraft_ids)

    output_path = DATA_DIR / "prediction_age_distribution.csv"
    write_distribution(distribution, output_path)
    print()
    print(f"Distribution detaillee ecrite dans : {output_path}")


if __name__ == "__main__":
    main()
